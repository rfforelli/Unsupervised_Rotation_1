add_files myproject_prj/solution1/syn/vhdl
synth_design -top myproject -part xcu200-fsgd2104-2-e
report_utilization -file vivado_synth.rpt