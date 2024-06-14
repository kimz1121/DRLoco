robot_ws_path="$HOME/robot_learn"
log_primitive_dir="logs_0613_final_only/mcp_dir_determine_45/seed1/models"
log_transfer_dir="DRLoco/logs_0614_transfer"
id="mcp_transfer_dir_determine_45_to_all"
cd $robot_ws_path
ptrhon_script="train_edit_06_mcp_naive_transfer_dir_all.py"
echo "now running $ptrhon_script"
# dir=225
python ./DRLoco/drloco/$ptrhon_script --vec_normalise true --id $id --seed 0 --logdir_primitive $log_primitive_dir --logdir_transfer $log_transfer_dir
