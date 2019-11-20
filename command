CUDA_VISIBLE_DEVICES=1 python train_image_classifier.py \
    --train_dir=/home/vip/qyr/data/car_color_data/slim_train \
    --dataset_name=mydata \
    --dataset_split_name=train \
    --dataset_dir=/home/vip/qyr/data/car_color_data/train_half_car \
    --model_name=resnet_v1_50 \
    --batch_size=32 \
    --attention_module=se_block \
    2>&1 | tee slim_resnet_50_color_half.log


CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=/home/vip/qyr/data/car_color_data/slim_train_ft \
    --dataset_dir=/home/vip/qyr/data/car_color_data/train_half_car \
    --dataset_name=mydata \
    --dataset_split_name=train \
    --model_name=resnet_v1_50 \
    --checkpoint_path=./resnet_v1_50.ckpt \
    --checkpoint_exclude_scopes=resnet_v1_50/logits \
    2>&1 | tee slim_resnet_50_color_half_ft.log


CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=/home/vip/qyr/data/car_color_data/slim_mobilenet_v1 \
    --dataset_dir=/home/vip/qyr/data/car_color_data/train_half_car \
    --dataset_name=mydata \
    --checkpoint_path=./mobile_ckpt/mobilenet_v1_1.0_224.ckpt \
    --dataset_split_name=train \
    --model_name=mobilenet_v1 \
    --checkpoint_exclude_scopes=MobilenetV1/Logits \
    2>&1 | tee slim_mobilenet_v1_color_half_ft.log


CUDA_VISIBLE_DEVICES=0 python train_image_classifier.py \
    --train_dir=/home/vip/qyr/data/car_color_data/slim_lenet \
    --dataset_dir=/home/vip/qyr/data/car_color_data/train_half_car \
    --dataset_name=mydata \
    --dataset_split_name=train \
    --model_name=lenet \
    2>&1 | tee slim_lenet_color_half_ft.log


CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=/home/vip/qyr/data/car_color_data/slim_train_ft \
    --dataset_dir=/home/vip/qyr/data/car_color_data/train_half_car \
    --dataset_name=mydata \
    --dataset_split_name=validation \
    --model_name=resnet_v1_50

INFO:tensorflow:Restoring parameters from /home/vip/qyr/data/car_color_data/slim_train_ft/model.ckpt-89992
INFO:tensorflow:Evaluation [1/6]
INFO:tensorflow:Evaluation [2/6]
INFO:tensorflow:Evaluation [3/6]
INFO:tensorflow:Evaluation [4/6]
INFO:tensorflow:Evaluation [5/6]
INFO:tensorflow:Evaluation [6/6]
eval/Accuracy[0.818333328]
eval/Recall_5[0.99333334]
INFO:tensorflow:Finished evaluation at 2019-11-19-11:24:24



python export_inference_graph.py \
  --alsologtostderr \
  --input_checkpoint=/Users/qiuyurui/Desktop/car_color/model.ckpt-89992 \
  --model_name=resnet_v1_50 \
  --num_classes=12 \
  --scope=resnet_v1_50 \
  --output_file=./inception_v3_inf_graph.pb

CUDA_VISIBLE_DEVICES=0 python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=/home/vip/qyr/data/car_color_data/slim_mobilenet_v1 \
    --dataset_dir=/home/vip/qyr/data/car_color_data/train_half_car \
    --dataset_name=mydata \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1


/home/vip/qyr/data/car_color_data/train_new_crop