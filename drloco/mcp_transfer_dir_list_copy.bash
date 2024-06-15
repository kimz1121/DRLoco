#$1 첫번째 파라미터를 의미

robot_ws_path="$HOME/robot_learn"
log_primitive_dir="DRLoco/logs_0615_transfer/NewRandomGoalAnt-v2/PPO_mcp_transfer_dir_list_to_all_00/seed0/models"
log_transfer_dir="DRLoco/logs_$1"
id="mcp_transfer_dir_list_to_all_00"
cd $robot_ws_path
ptrhon_script="train_edit_06_mcp_naive_transfer_dir_all.py"
echo "now running $ptrhon_script"
python ./DRLoco/drloco/$ptrhon_script --vec_normalise true --id $id --seed 0 --logdir_primitive $log_primitive_dir --logdir_transfer $log_transfer_dir 


