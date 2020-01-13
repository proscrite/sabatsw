import gwy
import gwyutils
import sys
filename = sys.argv[1]
 # '/Users/pabloherrero/sabat/stm_data/november19_experiments/QE1/2019-11-12-QE1_001.sxm'

f = open(filename, "r")
container = gwy.gwy_app_file_load(filename)
ids = gwy.gwy_app_data_browser_get_data_ids(container)

cons = gwy.gwy_app_data_browser_get_containers()
for c in cons:
  dfields = gwyutils.get_data_fields_dir(c)
  for key in dfields.keys():
    datafield = dfields[key]
    print(datafield.get_xreal())

gwy.gwy_app_data_browser_select_data_field(container, ids[0])
container['/%u/base/palette' %ids[0] ] = 'Gold'
gwy.gwy_process_func_run("level", container, gwy.RUN_INTERACTIVE)
gwy.gwy_process_func_run("scars_remove", container, gwy.RUN_INTERACTIVE)

gwy.gwy_file_save(container, '%s-%02u.png' %(basename, 0), gwy.RUN_NONINTERACTIVE)
