# Role Simulation: ML & DevOps Engineer — Multi-Cloud (AWS vs. Azure) Deployment Runbook

**Project**: Project 02 · Deep Learning + Computer Vision (Disease Detection System)  
**Owner Roles**: ML Engineer & DevOps/MLOps Engineer  
**Container Image**: `29omchavan/mlops-project-2-cnn-disease-classifier:latest`

---

## 1. Local & DVC Pipeline Reproduction Runbook

```bash
# 1. Create virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Reproduce the entire 4-stage DAG using DVC
dvc repro

# 3. Inspect DVC tracked metrics
dvc metrics show

# 4. Run automated unit & integration tests
pytest tests/ -v

# 5. Start FastAPI Computer Vision server on http://localhost:8080
python app.py
```

---

## 2. Multi-Cloud Comparison: AWS (ECR + EC2) vs. Azure (ACR + Web App for Containers)

As covered in Krish Naik's *End-to-End Deep Learning Project With Deployments Azure and AWS*, deploying the exact same Docker image across two cloud providers reveals key operational trade-offs:

| Architectural Dimension | AWS Deployment (`ECR + EC2 Self-Hosted Runner`) | Azure Deployment (`ACR / Docker Hub + Azure Web App`) |
| :--- | :--- | :--- |
| **Compute Abstraction** | **IaaS (Infrastructure-as-a-Service)**: You manage the Ubuntu VM, Docker daemon, security groups, and OS patches. | **PaaS (Platform-as-a-Service)**: Azure App Service manages the underlying VM, container orchestration, TLS, and restarts. |
| **Container Registry** | Amazon Elastic Container Registry (`AWS ECR`) | Azure Container Registry (`ACR`) or Docker Hub |
| **CI/CD Runner Pattern** | Requires registering a **GitHub Self-Hosted Runner** (`./config.sh && ./run.sh`) inside the EC2 instance. | Uses standard GitHub-hosted `ubuntu-latest` runner with `azure/webapps-deploy@v3` action or ACR webhook. |
| **Port & Networking** | Must explicitly open Inbound TCP Port `8080` in EC2 Security Group rules. | Automatically maps `WEBSITES_PORT=8080` to HTTPS (`https://<app-name>.azurewebsites.net`). |
| **Cost & Scaling** | Predictable hourly EC2 billing (`t3.medium`); manual Auto Scaling Group setup. | App Service Plan (`B1`/`P1v3`); built-in horizontal scale-out rules. |

---

## 3. Incident Post-Mortem & Troubleshooting Playbook

### Incident 1: Giant 2.5GB Docker Images Timing Out CI/CD Runners
* **Root Cause**: Default `pip install torch torchvision` on Linux pulls ~2.5GB of NVIDIA CUDA/cuDNN shared libraries even when deploying to CPU cloud instances.
* **Resolution**: Added `--extra-index-url https://download.pytorch.org/whl/cpu` to `requirements.txt`, shrinking the PyTorch wheel footprint by **>80%** (~180MB) and cutting Docker build/push times from 12 minutes to under 2 minutes.
