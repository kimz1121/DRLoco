#$1 첫번째 파라미터를 의미

robot_ws_path="$HOME/robot_learn"
log_primitive_dir="DRLoco/logs_0613_final_only/mcp_dir_determine_90/seed1/models"
log_transfer_dir="DRLoco/logs_$1"
id="mcp_transfer_dir_determine_90_to_135"
cd $robot_ws_path
ptrhon_script="train_edit_07_mcp_naive_transfer_dir_determine.py"
echo "now running $ptrhon_script"
dir=135
python ./DRLoco/drloco/$ptrhon_script --direction $dir --vec_normalise true --id $id --seed 0 --logdir_primitive $log_primitive_dir --logdir_transfer $log_transfer_dir
