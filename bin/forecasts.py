from exec_anaconda import exec_anaconda_or_die
from splunk_path_util import get_mltk_bin_path, get_spool_path

import sys

exec_anaconda_or_die()

sys.path.append(get_mltk_bin_path())

with open("/Users/gburgett/Downloads/frank.txt", "w") as ofile:
    ofile.write(str(sys.path))

import json
import time
import uuid
from io import StringIO
from collections import defaultdict
from datetime import datetime

import pandas as pd

from splunklib.client import connect
import splunk.Intersplunk as si
from statespace import StateSpaceForecast

ALGORITHMS = {
    "statespace": StateSpaceForecast
}

def result_reader_gen(result):
    for i, result_col in enumerate(result):
        if i == 0:
            yield result[result_col]
        else:
            yield {"_time": result_col, "value": float(result[result_col])}

def fit_result(result, options):
    result_gen = result_reader_gen(result)
    obj_id = next(result_gen)
    df = pd.DataFrame(result_gen)
    print(df[:70][["_time"]].values.dtype)
    algo_instance = ALGORITHMS[algo](options)

    if fit_type == "init":
        algo_instance.fit(df, options)
    else:
        algo_instance.partial_fit(df)

    if command_mode == "write":
        algo_instance.write_predictions(df, mode, forecast_k, conf_interval)
    else:
        algo_instance.display(df, mode, forecast_k)


def write_predictions(dataframe, object_id, index, marker=""):
    spool_filename = "%s_forecast.stash_new" % str(uuid.uuid1())
    summary_file_path = get_spool_path(spool_filename)
    with open(summary_file_path, "w") as ofile:
        ofile.write("***SPLUNK*** index=%s\n==##~~##~~  1E8N3D4E6V5E7N2T9 ~~##~~##==\n" % index)   
        for row in dataframe:
            result_dict = json.loads(row)["result"]
            ofile.write('%s object_id="%s" projected_date="%s" prediction="%f" upper%d="%f" lower%d="%f" %s\n==##~~##~~  1E8N3D4E6V5E7N2T9 ~~##~~##==\n' % (
                datetime.now(),
                result_dict[object_id],
                result_dict["projected_time"],
                float(result_dict["prediction"]),
                70,
                float(result_dict["lower70"]),
                70,
                float(result_dict["upper70"]),
                marker)
            )

if __name__ == "__main__":
    try:
        keywords, options = si.getKeywordsAndOptions()
        algo_options = {}
        algo_options["algo"] = algo = options.get('algo', 'statespace')
        algo_options["mode"] = command_mode = options.get('mode', None)
        algo_options["fit_type"] = fit_type = options.get('fit', 'init')
        algo_options["target_index"] = options.get('target_index', "summary")
        algo_options["forecast_k"] = options.get('forecast_k', 30)
        algo_options["conf_interval"] = options.get('conf_interval', 70)
        algo_options["feature_variables"] = ["_time", "value"]
        collection_name = options.get('collection', "ssf_default")

        if not command_mode:
            si.generateErrorResults('Requires "mode" field for applying initial or partial fit.')
            exit(0)

        results, dummyresults, settings = si.getOrganizedResults()
        print(options)
        sessionKey = settings.get("sessionKey", None)
        service = connect(token=sessionKey, owner="nobody", app="forecast")

        try:
            algo_options["collection"] = service.kvstore[collection_name]
        except Exception:
            si.generateErrorResults('"%s" collection does not exist or you do not have permissions to it.' % collection_name)
            exit(0)
        # collection.data.batch_save([{"_key": "item1", "somekey": 1, "otherkey": "foo"}])
        for result in results:
            fit_result(result, algo_options)

        output_results = []
        output_results.append({"status": "Completed Writing", "object_count": len(results), "index": target_index, "model": collection_name})
        si.outputResults(output_results)

    except Exception as e:
        import traceback
        stack = traceback.format_exc()
        si.generateErrorResults("Error '%s'. %s" % (e, stack))
