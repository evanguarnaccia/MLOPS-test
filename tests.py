import wandb
from wandb.errors import CommError
import dataikuapi
from dataikuapi import DSSClient
import pandas as pd
import os
import pytest
import urllib3


# Disable warnings for unverified HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Environment variables
DATAIKU_INSTANCE_URL = "https://dss-evan-sanofi.fe-aws.dkucloud-dev.com"
DATAIKU_API_KEY = "dkuaps-MdsgAu6ntCYFADNfNTsyGBks4gjJsd3U"
DATAIKU_PROJECT_KEY = "TUT_PROJECT_STANDARDS"
WANDB_API_KEY = "3c37e9da1f1145e2202f89d7223b45293643196b"

client = dataikuapi.DSSClient(DATAIKU_INSTANCE_URL, DATAIKU_API_KEY)
project = client.get_project(DATAIKU_PROJECT_KEY)

print(f"Connected to project: {DATAIKU_PROJECT_KEY}")

# --- Retrieve authentication info with secrets ---
#auth_info = client.get_auth_info(with_secrets=False)
#secret_value = None
#for secret in auth_info.get("secrets", []):
#    if secret.get("key") == "wandbcred":
#        secret_value = "3c37e9da1f1145e2202f89d7223b45293643196b"
#        break
#if not secret_value:
#    raise Exception("Secret 'wandbcred' not found")

# --- W&B login / API ---
wandb.login(key=WANDB_API_KEY)
#wandb.login(key=secret_value)

api = wandb.Api()



# --- Project and saved models from Dataiku ---
project = client.get_project(DATAIKU_PROJECT_KEY)
saved_model_ids = [sm['id'] for sm in project.list_saved_models()]

if not saved_model_ids:
    print("W&B DEBUG: No saved models found in Dataiku.")
else:
    # --- Collect ALL W&B model artifacts once ---
    artifacts = []
    try:
        for collection in api.registries().collections():
            for artifact in collection.artifacts():
                if artifact.type and artifact.type.lower() == "model":
                    artifacts.append({
                        "collection": collection.name,
                        "artifact": artifact.source_name,   # display name, e.g. "dataiku-<sm>-<ver>:v0"
                        "path": artifact.qualified_name     # e.g. "entity/project/artifact:version"
                    })
    except CommError as e:
        raise RuntimeError(f"Failed to list W&B artifacts: {e}")

    artifact_names = [{'name': a['artifact'], 'path': a['path']} for a in artifacts]

    any_published = False  # overall flag across all models

    # --- Iterate each saved model and check against W&B ---
    for sm in saved_model_ids:
        print("✅----- Checking Model(s) in Dataiku -----")
        print(f"Dataiku Current Saved Model : {sm}")

        model = project.get_saved_model(sm)
        active_id = model.get_active_version()['id']
        _ = model.get_version_details(active_id)  # kept in case details needed later

        model_identifier = f"dataiku-{sm}-{active_id}"
        print(f"Dataiku Model Identifier  : {model_identifier}")
        print("----- Checking if Model exists in W&B -----")

        # Filter W&B artifacts that match this model
        candidate_artifacts = [a for a in artifact_names if model_identifier in a['name']]

        if not candidate_artifacts:
            print("⚠️  No published W&B artifacts found for this model.\n")
            continue  # go check next saved model

        # We found at least one published model for this Dataiku model
        any_published = True
        for art in candidate_artifacts:
            wb_name_full = art['name']              # e.g. "dataiku-4wUI1vp8-1702915444643:v0"
            if ":" in wb_name_full:
                wb_name_base, wb_version = wb_name_full.split(":", 1)
            else:
                wb_name_base, wb_version = wb_name_full, None

            print("✅ Published Model Found in W&B")
            print(f"   Full Artifact Name : {wb_name_full}")
            print(f"   Base Name          : {wb_name_base}")
            print(f"   W&B Version        : {wb_version}")
            print(f"   Registry Path      : {art['path']}")
            print("------------------------------")

    if not any_published:
        print(" 🛑W&B DEBUG: No models are published to W&B for any saved models.")

