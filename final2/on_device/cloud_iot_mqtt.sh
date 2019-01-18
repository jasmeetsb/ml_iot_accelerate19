#!/usr/bin/env bash
# Script executes the csv to json parsing and send the telemetry data to Google Cloud IoT Core via MQTT
python csv2json.py
wget -N https://pki.goog/roots.pem
python cloud_iot_mqtt.py --registry_id=iot-edgeml-demo --project_id=iot-edge-ml --device_id=iot-ml-dev1 --algorithm=RS256 --private_key_file=/etc/gcloud-edge/certificates/rsa_private.pem
rm ./output.json
