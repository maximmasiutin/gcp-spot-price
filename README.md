# GCP Spot VM Price Finder

Find the cheapest Google Cloud spot instances across all regions.

Queries two GCP APIs and combines the results:

1. **Cloud Billing Catalog API** -- spot pricing per vCPU-hour and per GB-hour, grouped by machine family and region
2. **Compute Engine API** -- machine type specs (vCPUs, memory) and zone availability

Total spot cost = `(vCPUs * cpu_rate) + (memory_GB * ram_rate)`. Rates vary by machine family (e2, n2, c3, ...) and region.

## Prerequisites

**Python 3.14+** and a GCP project with Compute Engine API enabled.

```
pip install -r requirements.txt
```

Dependencies: `google-cloud-billing`, `google-cloud-compute`, `tabulate`.

**Authentication** -- one of:

```bash
# Option 1: Application Default Credentials (recommended)
gcloud auth application-default login

# Option 2: Service account key
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

The authenticated account needs:
- `cloudbilling.services.list` + `cloudbilling.skus.list` (roles/billing.viewer or similar)
- `compute.machineTypes.list` on the target project (roles/compute.viewer)

## Usage

```
python gcp-spot-price.py --project PROJECT [options]
```

`--project` is required -- any GCP project where Compute Engine API is enabled.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-c, --cpu N` | Minimum vCPUs | 2 |
| `-m, --mem N` | Minimum memory (GB) | 0 |
| `--region REGION` | Single region or `all` | all |
| `--family LIST` | Include only these families (comma-separated) | all |
| `--exclude-family LIST` | Exclude these families | none |
| `--exclude-regions LIST` | Exclude these regions | none |
| `--cheapest` | Show cheapest zone per machine type | off |
| `--single` | Show only the single cheapest result | off |
| `--top N` | Show top N results | all |
| `--no-shared-core` | Exclude micro/small/medium types | off |
| `--sort-by-core` | Sort by per-vCPU price instead of total | off |

### Examples

**Single cheapest 4+ vCPU spot instance globally:**
```bash
python gcp-spot-price.py --project my-project --cpu 4 --single
```

**Top 20 cheapest 64+ vCPU, 256+ GB RAM instances:**
```bash
python gcp-spot-price.py --project my-project --cpu 64 --mem 256 --top 20
```

**Cheapest per machine type in us-central1:**
```bash
python gcp-spot-price.py --project my-project --cpu 8 --region us-central1 --cheapest
```

**Only N2 and C3 families, sorted by per-core price:**
```bash
python gcp-spot-price.py --project my-project --cpu 16 --family n2,c3 --sort-by-core
```

**Exclude GPU families and specific regions:**
```bash
python gcp-spot-price.py --project my-project --cpu 4 --exclude-family a2,a3,g2 --exclude-regions us-east1,asia-south1
```

### Output

```
| Machine Type     | Zone              | $/hr   | $/vCPU/hr | vCPUs | RAM GB |
|------------------+-------------------+--------+-----------+-------+--------|
| e2-standard-4    | us-central1-a     | 0.0401 | 0.010025  |     4 |   16.0 |
| n2-standard-4    | us-central1-a     | 0.0486 | 0.012150  |     4 |   16.0 |
| c3-standard-4    | us-central1-a     | 0.0510 | 0.012750  |     4 |   16.0 |
```

## How It Works

1. Fetches all Compute Engine spot pricing SKUs from the Billing Catalog API (filters `usage_type == "Preemptible"`)
2. Parses SKU descriptions to extract machine family and component (Core/Ram)
3. Fetches machine types via Compute Engine `aggregated_list` across all zones
4. For each machine type: looks up spot rates by family + region, computes `total = vCPUs * cpu_rate + memory * ram_rate`
5. Sorts and displays results

Supported families: e2, n1, n2, n2d, n4, c2, c2d, c3, c3d, c4, c4a, m1, m2, m3, t2a, t2d, a2, a3, h3, g2, z3. New families are auto-detected via description parsing.

## See Also

- [AWS Spot Instance Finder](https://github.com/maximmasiutin/aws-spot-instance-finder) -- Find cheapest AWS EC2 spot instances across all regions
- [Azure Spot VM Price Finder](https://github.com/maximmasiutin/azure-scripts#cost-optimization-scripts) -- Find cheapest Azure spot instances with vm-spot-price.py
