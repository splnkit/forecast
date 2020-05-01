import json
import uuid
from datetime import datetime

def write_predictions(dataframe, object_id, index):
    spool_filename = "%s_forecast.stash_new" % str(uuid.uuid1())
    summary_file_path = "/Users/gburgett/Downloads/%s" % spool_filename
    with open(summary_file_path, "w") as ofile:
        ofile.write("***SPLUNK*** index=%s\n==##~~##~~  1E8N3D4E6V5E7N2T9 ~~##~~##==\n" % index)   
        for row in dataframe:
            result_dict = json.loads(row)["result"]
            ofile.write('%s object_id="%s" projected_date="%s" prediction="%f" upper%d="%f" lower%d="%f" search_id="%s"\n==##~~##~~  1E8N3D4E6V5E7N2T9 ~~##~~##==\n' % (
                datetime.now(),
                result_dict[object_id],
                result_dict["projected_time"],
                float(result_dict["prediction"]),
                70,
                float(result_dict["lower70"]),
                70,
                float(result_dict["upper70"]),
                "sid")
            )

with open("/Users/gburgett/Downloads/forecast_output.json","rb") as infile:
	write_predictions(infile, "object_id", "ssf_summary")