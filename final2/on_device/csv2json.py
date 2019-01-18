import csv
import json

csvfile = open('data.csv', 'r')
jsonfile = open('output.json', 'w')

fieldnames = ('infer_time','device_id','can_detect','Orientation','Score','Counter')
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')
