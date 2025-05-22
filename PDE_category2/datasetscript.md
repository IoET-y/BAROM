
# convdiff
python generate_integral_feedback_datasets.py --dataset_type convdiff --num_samples 5000 --nx 64 --nt 300 --T 2.0 --output_dir ./datasets_new_feedback --filename_suffix _v1

# Generate Reaction-Diffusion with Neumann Integral Feedback
python generate_feedbackdata.py --dataset_type reaction_diffusion_neumann_feedback --num_samples 5000 --nx 64 --nt 300 --T 2.0 --output_dir ./datasets_integral_feedback --filename_suffix _v1

# Generate Heat Equation with Non-linear Feedback Gain
python generate_feedbackdata.py --dataset_type heat_nonlinear_feedback_gain --num_samples 5000 --nx 64 --nt 300 --num_controls 1 --T 2.0 --output_dir ./datasets_integral_feedback --filename_suffix _v1