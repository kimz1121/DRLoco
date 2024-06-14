robot_ws_path="$HOME/robot_learn"
log_primitive_dir="logs_0613_final_only/mcp_dir_list/seed1/models"
log_transfer_path="DRLoco/logs_0614_transfer"
id="mcp_transfer_dir_list_to_all_00_tests_copy"
cd $robot_ws_path
ptrhon_script="train_edit_06_mcp_naive_transfer_dir_all.py"
echo "now running $ptrhon_script"
python ./DRLoco/drloco/$ptrhon_script --vec_normalise true --id $id --seed 0 --logdir_primitive $log_primitive_dir --logdir_transfer $log_transfer_path --eval_freq 1000000


