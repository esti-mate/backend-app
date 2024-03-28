
from google.cloud import aiplatform
from constants import GCP_LOCATION, GCP_PROJECT_ID
from flask import jsonify


aiplatform.init(location=GCP_LOCATION,project=GCP_PROJECT_ID)


def start_train_job(display_name, machine_type,train_ds_path ,org_id, batch_size_ratio, epoch, sequence_length):
    try:
        job = aiplatform.CustomPythonPackageTrainingJob(
            display_name=display_name,
            python_package_gcs_uri="gs://estimate-bucket/model-app-0.3.tar.gz",
            python_module_name="trainer.app",
            container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest",
            staging_bucket="finished-models",
        )

        job.run(
            replica_count=1,
            
            machine_type=machine_type,
            service_account="450895866029-compute@developer.gserviceaccount.com",
            # accelerator_count=1,
            # accelerator_type="NVIDIA_TESLA_T4",
            sync=False,

            # ds_file_name = sys.argv[1]
            # org_id = sys.argv[2]
            # batch_size_ratio = sys.argv[3]
            # epoch = sys.argv[4]
            # sequence_length = sys.argv[5]

            args=[train_ds_path,org_id,batch_size_ratio,epoch,sequence_length] # pass the arguments to the trainer app
        )

        job.wait_for_resource_creation()
        return jsonify({"name": str(job.name) , "resource_name":job.resource_name }), 200
    except Exception as e:
        print("error ::: ", e)
        return jsonify({"error": str(e)}), 500



# def get_train_job_status(custom_job_id):
#     # TODO : Make this to work with the custom job id
#     client_options = {"api_endpoint": GCP_LOCATION+"-aiplatform.googleapis.com"}
#     client = aiplatform.gapic.JobServiceClient(client_options=client_options)
#     name = client.c(project=GCP_PROJECT_ID, location=GCP_LOCATION, custom_job=custom_job_id)
#     response = client.get_pipeline_job(name=name)

#     return response