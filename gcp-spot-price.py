#!/usr/bin/env python
# pylint: disable=invalid-name
"""GCP Spot VM Price Finder - Find cheapest Google Cloud spot instances.

Copyright 2026 Maxim Masiutin. All rights reserved.

Queries the Google Cloud Billing Catalog API for spot (preemptible) pricing
and the Compute Engine API for machine type specifications, then combines
them to find the cheapest spot VMs across regions.

GCP spot pricing model:
  Total cost = (vCPUs * per-vCPU-hour rate) + (memory_GB * per-GB-hour rate)
  Rates vary by machine family (e2, n2, c3, ...) and region.

Prerequisites:
  pip install -r requirements.txt
  gcloud auth application-default login

Usage:
  python gcp-spot-price.py --project my-project --cpu 4
  python gcp-spot-price.py --project my-project --cpu 64 --mem 256 --single
  python gcp-spot-price.py --project my-project --cpu 8 --region us-central1
  python gcp-spot-price.py --project my-project --cpu 16 --family n2,c3 --cheapest
  python gcp-spot-price.py --project my-project --cpu 4 --top 20
"""

import argparse
import dataclasses
import re
import sys
import warnings
from collections import defaultdict

from google.api_core import exceptions as gapi_exceptions
from google.cloud import billing_v1, compute_v1
from tabulate import tabulate


COMPUTE_ENGINE_SERVICE = "services/6F81-5844-456A"

BILLING_FAMILY_MAP: dict[str, str] = {
    "n1 predefined": "n1",
    "compute optimized": "c2",
    "memory-optimized": "m2",
    "n2d": "n2d",
    "c2d": "c2d",
    "c3d": "c3d",
    "c4a": "c4a",
    "t2a": "t2a",
    "t2d": "t2d",
    "e2": "e2",
    "n2": "n2",
    "n4": "n4",
    "c2": "c2",
    "c3": "c3",
    "c4": "c4",
    "m1": "m1",
    "m2": "m2",
    "m3": "m3",
    "a2": "a2",
    "a3": "a3",
    "h3": "h3",
    "g2": "g2",
    "z3": "z3",
}

SHARED_CORE_SUFFIXES = frozenset({"micro", "small", "medium"})
SHARED_CORE_NAMES = frozenset({"f1-micro", "g1-small"})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="GCP Spot VM Price Finder",
        epilog="Find cheapest spot VMs across Google Cloud regions.",
    )
    p.add_argument(
        "--project", required=True, help="GCP project ID (for machine type lookup)"
    )
    p.add_argument(
        "-c", "--cpu", type=int, default=2, help="Minimum vCPUs (default: 2)"
    )
    p.add_argument(
        "-m", "--mem", type=float, default=0, help="Minimum memory in GB (default: 0)"
    )
    p.add_argument(
        "--region", default="all", help="GCP region or 'all' (default: all)"
    )
    p.add_argument(
        "--family",
        default=None,
        help="Machine family filter, comma-separated (e.g., n2,c3,e2)",
    )
    p.add_argument(
        "--exclude-family",
        default=None,
        help="Exclude machine families, comma-separated (e.g., a2,a3,g2)",
    )
    p.add_argument(
        "--cheapest",
        action="store_true",
        help="Show cheapest zone per machine type",
    )
    p.add_argument(
        "--single",
        action="store_true",
        help="Show only the single cheapest result",
    )
    p.add_argument(
        "--top", type=int, default=0, help="Show top N results (default: show all)"
    )
    p.add_argument(
        "--no-shared-core",
        action="store_true",
        help="Exclude shared-core types (micro, small, medium)",
    )
    p.add_argument(
        "--sort-by-core",
        action="store_true",
        help="Sort by per-vCPU price instead of total price",
    )
    p.add_argument(
        "--exclude-regions",
        default=None,
        help="Exclude regions, comma-separated (e.g., us-east1,asia-south1)",
    )
    return p.parse_args()


def extract_family_and_component(
    description: str,
) -> tuple[str | None, str | None]:
    """Extract machine family and component from billing SKU description.

    Examples:
        "Spot Preemptible N2 Instance Core running in Americas" -> ("n2", "cpu")
        "Spot Preemptible E2 Instance Ram running in EMEA" -> ("e2", "ram")
        "Spot Preemptible N1 Predefined Instance Core ..." -> ("n1", "cpu")
    """
    m = re.search(
        r"Spot Preemptible (.+?) Instance (Core|Ram) running in", description
    )
    if not m:
        return None, None

    raw = m.group(1).strip().lower()
    component = "cpu" if m.group(2) == "Core" else "ram"

    for key in sorted(BILLING_FAMILY_MAP, key=len, reverse=True):
        if raw == key:
            return BILLING_FAMILY_MAP[key], component

    if clean := re.sub(r"[^a-z0-9]", "", raw):
        return clean, component

    return None, None


def _diagnose_billing_error(exc: Exception) -> str:
    """Return a human-readable diagnosis for Billing Catalog API errors."""
    msg = str(exc)
    if isinstance(exc, gapi_exceptions.Unauthenticated):
        return (
            "Authentication failed for the Cloud Billing Catalog API.\n"
            "Fix: run 'gcloud auth application-default login' and retry."
        )
    if isinstance(exc, gapi_exceptions.PermissionDenied):
        return (
            "Permission denied accessing the Cloud Billing Catalog API.\n"
            "The Billing API is public, so this usually means your credentials\n"
            "are invalid or expired.\n"
            "Fix: run 'gcloud auth application-default login' and retry."
        )
    if isinstance(exc, gapi_exceptions.ResourceExhausted):
        return (
            "Quota exceeded on the Cloud Billing Catalog API.\n"
            "Your credentials lack a quota project, so requests count against\n"
            "a shared default quota which is easily exhausted.\n"
            "Fix: run 'gcloud auth application-default set-quota-project PROJECT_ID'\n"
            "     (use any project where you have serviceusage.services.use permission)."
        )
    if "quota" in msg.lower():
        return (
            "Quota-related error from the Billing API.\n"
            "Fix: run 'gcloud auth application-default set-quota-project PROJECT_ID'\n"
            "     to assign a quota project to your application-default credentials."
        )
    return f"Unexpected Billing API error: {msg}"


def fetch_spot_pricing() -> dict[tuple[str, str], dict[str, float]]:
    """Fetch spot pricing from Cloud Billing Catalog API.

    Returns: {(family_prefix, region): {"cpu": price, "ram": price}}
    """
    print("Fetching spot pricing from Billing Catalog API...", flush=True)

    try:
        client = billing_v1.CloudCatalogClient()
    except Exception as exc:
        print(f"Error: cannot create Billing Catalog client: {exc}", file=sys.stderr)
        print(
            "Fix: ensure google-cloud-billing is installed and credentials are set up.\n"
            "     Run: pip install google-cloud-billing\n"
            "     Run: gcloud auth application-default login",
            file=sys.stderr,
        )
        sys.exit(1)

    prices: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"cpu": 0.0, "ram": 0.0}
    )

    sku_count = 0
    try:
        for sku in client.list_skus(parent=COMPUTE_ENGINE_SERVICE):
            if sku.category.usage_type != "Preemptible":
                continue
            if sku.category.resource_family != "Compute":
                continue

            family, component = extract_family_and_component(sku.description)
            if not family or not component:
                continue

            if not sku.pricing_info:
                continue
            expr = sku.pricing_info[0].pricing_expression
            if not expr or not expr.tiered_rates:
                continue

            rate = expr.tiered_rates[0]
            price = float(rate.unit_price.units) + float(rate.unit_price.nanos) / 1e9
            if price <= 0:
                continue

            for region in sku.service_regions:
                prices[(family, region)][component] = price
                sku_count += 1
    except Exception as exc:
        print(f"Error fetching spot pricing: {exc}", file=sys.stderr)
        print(_diagnose_billing_error(exc), file=sys.stderr)
        sys.exit(1)

    families = {k[0] for k in prices}
    regions = {k[1] for k in prices}
    print(
        f"  Found pricing for {len(families)} families"
        f" in {len(regions)} regions ({sku_count} entries)."
    )
    return dict(prices)


def _diagnose_compute_error(exc: Exception, project: str) -> str:
    """Return a human-readable diagnosis for Compute Engine API errors."""
    msg = str(exc)
    if isinstance(exc, gapi_exceptions.Unauthenticated):
        return (
            "Authentication failed for the Compute Engine API.\n"
            "Fix: run 'gcloud auth application-default login' and retry."
        )
    if isinstance(exc, gapi_exceptions.Forbidden):
        if "has not been used" in msg or "it is disabled" in msg:
            return (
                f"The Compute Engine API is not enabled in project '{project}'.\n"
                "Fix: enable it by running:\n"
                f"     gcloud services enable compute.googleapis.com"
                f" --project={project}\n"
                "Wait a minute after enabling, then retry.\n"
                "Alternatively, use a different project that already has the\n"
                "Compute Engine API enabled (pass --project=OTHER_PROJECT)."
            )
        return (
            f"Permission denied on project '{project}'.\n"
            "Your account does not have permission to list machine types.\n"
            "Fix: ensure you have at least the 'Compute Viewer' role\n"
            "     (roles/compute.viewer) on the project, or use a project\n"
            "     where you have access."
        )
    if isinstance(exc, gapi_exceptions.NotFound):
        return (
            f"Project '{project}' was not found.\n"
            "Fix: verify the project ID is correct. You can list your projects\n"
            "     with: gcloud projects list"
        )
    if isinstance(exc, gapi_exceptions.ResourceExhausted):
        return (
            "Quota exceeded on the Compute Engine API.\n"
            "Fix: run 'gcloud auth application-default set-quota-project PROJECT_ID'\n"
            "     to set a quota project for your credentials, or wait and retry."
        )
    return (
        f"Unexpected Compute Engine API error: {msg}\n"
        "General troubleshooting:\n"
        "  1. Run: gcloud auth application-default login\n"
        f"  2. Verify project exists: gcloud projects describe {project}\n"
        "  3. Check API is enabled: gcloud services list"
        f" --project={project} --filter=compute"
    )


def fetch_machine_types(
    project: str, region_filter: str | None = None
) -> list[tuple[str, str, int, float]]:
    """Fetch machine types from Compute Engine API via aggregated_list.

    Returns: [(name, zone, vcpus, memory_gb), ...]
    """
    client = compute_v1.MachineTypesClient()
    types: list[tuple[str, str, int, float]] = []

    print("Fetching machine types from Compute Engine API...", flush=True)

    try:
        for zone_scope, scoped_list in client.aggregated_list(project=project):
            if not scoped_list.machine_types:
                continue

            zone_name = zone_scope.replace("zones/", "")
            region = zone_name.rsplit("-", 1)[0]

            if region_filter and region != region_filter:
                continue

            for mt in scoped_list.machine_types:
                types.append(
                    (mt.name, zone_name, mt.guest_cpus, mt.memory_mb / 1024.0)
                )

    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Error fetching machine types: {exc}", file=sys.stderr)
        print(_diagnose_compute_error(exc, project), file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(types)} machine type/zone combinations.")
    return types


def is_shared_core(name: str) -> bool:
    """Check if a machine type is a shared-core instance."""
    if name in SHARED_CORE_NAMES:
        return True
    parts = name.split("-")
    return len(parts) >= 2 and parts[-1] in SHARED_CORE_SUFFIXES


@dataclasses.dataclass(frozen=True, slots=True)
class Filters:
    """Filter criteria for machine type selection."""

    families: set[str] | None
    exclude_families: set[str]
    exclude_regions: set[str]


def _parse_csv_set(value: str | None) -> set[str]:
    """Parse a comma-separated string into a lowercase set."""
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",")}


def _print_header(args: argparse.Namespace) -> None:
    """Print run configuration header."""
    if args.single:
        mode = "single cheapest"
    elif args.cheapest:
        mode = "cheapest per type"
    else:
        mode = "all"
    parts = [f"Mode: {mode} | Min vCPUs: {args.cpu}"]
    if args.mem:
        parts.append(f" | Min RAM: {args.mem:.0f} GB")
    if args.region != "all":
        parts.append(f" | Region: {args.region}")
    if args.family:
        parts.append(f" | Families: {args.family}")
    print("".join(parts))
    print()


def _compute_costs(
    machine_types: list[tuple[str, str, int, float]],
    spot_prices: dict[tuple[str, str], dict[str, float]],
    args: argparse.Namespace,
    filters: Filters,
) -> list[list]:
    """Compute spot costs for each matching machine type."""
    results: list[list] = []

    for name, zone, vcpus, memory_gb in machine_types:
        if vcpus < args.cpu:
            continue
        if args.mem and memory_gb < args.mem:
            continue
        if args.no_shared_core and is_shared_core(name):
            continue

        family = name.split("-")[0]
        if filters.families and family not in filters.families:
            continue
        if family in filters.exclude_families:
            continue

        region = zone.rsplit("-", 1)[0]
        if region in filters.exclude_regions:
            continue

        pricing = spot_prices.get((family, region))
        if not pricing or not pricing["cpu"]:
            continue

        total = (vcpus * pricing["cpu"]) + (memory_gb * pricing["ram"])
        if total <= 0:
            continue

        per_core = total / vcpus
        results.append([name, zone, total, per_core, vcpus, round(memory_gb, 1)])

    return results


def _filter_and_display(results: list[list], args: argparse.Namespace) -> int:
    """Sort, filter, and display results. Returns exit code."""
    if not results:
        print("\nNo matching spot instances found.")
        return 1

    sort_idx = 3 if args.sort_by_core else 2
    results.sort(key=lambda x: x[sort_idx])

    if args.single:
        results = results[:1]
    elif args.cheapest:
        seen: set[str] = set()
        filtered = []
        for r in results:
            if r[0] not in seen:
                seen.add(r[0])
                filtered.append(r)
        results = filtered

    if args.top > 0:
        results = results[: args.top]

    display = [
        [r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.6f}", r[4], r[5]]
        for r in results
    ]

    print(f"\n{len(display)} result(s):\n")
    print(
        tabulate(
            display,
            headers=["Machine Type", "Zone", "$/hr", "$/vCPU/hr", "vCPUs", "RAM GB"],
            tablefmt="orgtbl",
        )
    )
    print()
    return 0


def main() -> int:
    """Find and display cheapest GCP spot VM instances."""
    warnings.filterwarnings("ignore", message=".*without a quota project.*")
    args = parse_args()

    print("GCP Spot VM Price Finder")
    _print_header(args)

    filters = Filters(
        families=_parse_csv_set(args.family) or None,
        exclude_families=_parse_csv_set(args.exclude_family),
        exclude_regions=_parse_csv_set(args.exclude_regions),
    )

    spot_prices = fetch_spot_pricing()
    region_arg = args.region if args.region != "all" else None
    machine_types = fetch_machine_types(args.project, region_arg)

    print("Calculating spot prices...", flush=True)
    results = _compute_costs(machine_types, spot_prices, args, filters)
    return _filter_and_display(results, args)


if __name__ == "__main__":
    sys.exit(main())
