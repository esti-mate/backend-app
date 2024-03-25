from flask import Flask, request, jsonify
from services.start_train_job import start_train_job as train
from services.run_prediction import get_predictions as predict
from jiraone import LOGIN , endpoint
from services.cloud_storage import upload_to_gcp
from constants import JIRA_EMAIL, JIRA_TOKEN ,JIRA_URL
from utils import get_random_string
import csv
import json
import os
app = Flask(__name__)


user = "email"
password = "token"
link = "https://yourinstance.atlassian.net"
LOGIN(user=JIRA_EMAIL, password=JIRA_TOKEN, url=JIRA_URL)
LOGIN.api=2

@app.route("/get-predictions", methods=["POST"])
def get_predictions():

    """Get predictions for a given organization ID and text."""

    # Ensure that request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validate presence of organizationId and text
    if "organizationId" not in data or "text" not in data:
        return jsonify({"error": "Missing 'organizationId' or 'text' in request"}), 400

    res = predict(organization_id=data["organizationId"], text=data["text"])

    return res


@app.route("/start-train-job", methods=["POST"])
def start_train_job():

    """Start a training job for a given display name, machine type, and training dataset path."""

    req = request.get_json()
    display_name = req["display_name"] 
    machine_type = req["machine_type"]
    train_ds_path = req["train_ds_path"]

    return train(display_name=display_name, machine_type=machine_type, train_ds_path=train_ds_path)

@app.route("/export-csv", methods=["POST"])
def export_csv():

    dataset_file_path = get_random_string() +".csv"
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    
    # Validate presence of jql
    if "jql" not in data:
        return jsonify({"error": "Missing 'jql' in request"}), 400
    
    # Connect to JiraOne
    issues=[]
    start_at=0
    while True:
        print("Requesting issues...starting_At",start_at)
        response = LOGIN.get(endpoint.search_issues_jql(query=data["jql"],start_at=start_at , max_results=100))
        response =  json.loads(response.text)
        issue_result =  response["issues"]
        

        start_at += len(issue_result) 
        issues.extend(response["issues"])

        if len(issues) == response["total"]:
            break

    # Write issues to CSV file
    with open(dataset_file_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["issuekey", "title", "description", "storypoint"])
        for issue in issues:
            writer.writerow(
                [
                    issue["key"],
                    issue["fields"]["summary"],
                    issue["fields"]["description"]
                ]
            )

    file_path , folder_path = upload_to_gcp("estimate-bucket",dataset_file_path,"datasets")
    os.remove(dataset_file_path)

    return jsonify({"gs_path":file_path  })
    
    # return jsonify({"message": "CSV file exported successfully"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3003)
