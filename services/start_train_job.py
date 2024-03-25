from google.cloud import aiplatform
from constants import GCP_LOCATION, GCP_PROJECT_ID
from flask import jsonify


aiplatform.init(location=GCP_LOCATION,project=GCP_PROJECT_ID)


def start_train_job(display_name, machine_type,train_ds_path):
    try:
        job = aiplatform.CustomPythonPackageTrainingJob(
            display_name=display_name,
            python_package_gcs_uri="gs://estimate-bucket/model-app-0.2.tar.gz",
            python_module_name="trainer.app",
            container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
            staging_bucket="finished-models"
        )
        
        job.run(
            replica_count=1,
            machine_type=machine_type,
            service_account="450895866029-compute@developer.gserviceaccount.com",
            # accelerator_count=8,
            # accelerator_type="",
            sync=False,
            args=[''] # pass the arguments to the trainer app
        )

        job.wait_for_resource_creation()
        return jsonify({"jobid": str(job.resource_name)}), 200
    except Exception as e:
        print("error ::: ", e)
        return jsonify({"error": str(e)}), 500
