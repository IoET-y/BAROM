python generate_datasets.py --dataset advection --num_samples 10000 --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full
python generate_datasets.py --dataset euler --num_samples 10000  --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full
python generate_datasets.py --dataset burgers --num_samples 10000  --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full
python generate_datasets.py --dataset darcy --num_samples 10000  --nx 128 --nt 600 --T 2 --num_controls 0 --output_dir ../datasets_full