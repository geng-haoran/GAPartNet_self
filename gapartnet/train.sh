CUDA_VISIBLE_DEVICES=7 \
python train.py fit -c gapartnet.yaml \
--model.init_args.ckpt ckpt/sem_seg_accu_82.7.ckpt \
--model.init_args.debug True

CUDA_VISIBLE_DEVICES=5,7 \
python train.py fit -c gapartnet.yaml \
--model.init_args.ckpt ckpt/offset_work_somehow.ckpt \
--model.init_args.debug True


CUDA_VISIBLE_DEVICES=3 \
python train.py test -c gapartnet.yaml \
--model.init_args.ckpt ckpt/offset_work_somehow.ckpt \
--model.init_args.debug True