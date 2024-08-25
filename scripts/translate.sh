python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist/config.yaml --output_zarr /datastore/cam5/nat_hist_to_hadgem3_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist/checkpoints/gen_00160000.pt --a2b 0 --seed 98876

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-4channels/config.yaml --output_zarr /datadrive/cam5/nat_hist_to_hadgem3_4ch_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-4channels/checkpoints/gen_00110000.pt --x2x b2a --seed 9725432

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v3/config.yaml --output_zarr /datadrive/cam5/nat_hist_to_hadgem3_v3_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v3/checkpoints/gen_00550000.pt --x2x b2a --seed 237897

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/config.yaml --output_zarr /datadrive/cam5/nat_hist_to_hadgem3_v5_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/checkpoints/gen_00069000.pt --x2x b2a --seed 202002111018

python translate.py --config ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/config.yaml --output_zarr /datadrive/hadgem3/nat_hist_to_hadgem3_v5.1_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_cam5_nat-hist-v5/checkpoints/gen_00056000.pt --x2x a2b --seed 202015121437


# hadgem to era global
python translate.py --config ~/model_outputs/outputs/global_hadgem3_to_era5/config.yaml --output_zarr /datadrive/hadgem3/all_hist_to_era5_zarr --checkpoint ~/model_outputs/outputs/global_hadgem3_to_era5/checkpoints/gen_00066000.pt  --x2x b2a --seed 202002081204

python translate.py --config ~/model_outputs/outputs/global_hadgem3_to_era5/config.yaml --output_zarr /datadrive/era5/all_hist_to_hadgem3_zarr --checkpoint ~/model_outputs/outputs/global_hadgem3_to_era5/checkpoints/gen_00066000.pt  --x2x a2b --seed 202002081204

# hadgem to era monsoon

python translate.py --config ~/model_outputs/outputs/hadgem3_to_era5_monsoon/config.yaml --output_zarr /datadrive/era5/monsoon_to_hadgem3_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_era5_monsoon/checkpoints/gen_00110000.pt  --x2x a2b --seed 202002091130

python translate.py --config ~/model_outputs/outputs/hadgem3_to_era5_monsoon/config.yaml --output_zarr /datadrive/hadgem3/monsoon_to_era5_zarr --checkpoint ~/model_outputs/outputs/hadgem3_to_era5_monsoon/checkpoints/gen_00110000.pt  --x2x b2a --seed 202002091130

# new start monsoon

python translate.py --config ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/config.yaml --output_zarr /datadrive/era5/v8_monsoon_to_hadgem --checkpoint ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/checkpoints/gen_00040000.pt  --x2x a2b --seed 202108261100

python translate.py --config ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/config.yaml --output_zarr /datadrive/hadgem3/v8_monsoon_to_era5 --checkpoint ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/checkpoints/gen_00040000.pt  --x2x b2a --seed 202108261100


python translate.py --config ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/config.yaml --output_zarr /datadrive/era5/v8.1_monsoon_to_hadgem --checkpoint ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/checkpoints/gen_00100000.pt  --x2x a2b --seed 202109061100

python translate.py --config ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/config.yaml --output_zarr /datadrive/hadgem3/v8.1_monsoon_to_era5 --checkpoint ~/model_outputs/outputs/v8_monsoon_hadgem_to_era5/checkpoints/gen_00100000.pt  --x2x b2a --seed 202109061100



python translate.py --config ~/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/config.yaml --output_zarr /datadrive/hadgem3/v8.2_monsoon_to_era5 --checkpoint ~/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/checkpoints/gen_00022000.pt  --x2x b2a --seed 202109061100

python translate.py --config /home/s1205782/geos-fulton/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/config.yaml --output_zarr /home/s1205782/geos-fulton/datadrive/hadgem3/v8.2_monsoon_to_era5_40k --checkpoint /home/s1205782/geos-fulton/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/checkpoints/gen_00040000.pt  --x2x b2a --seed 202109061100

python translate.py --config ~/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/config.yaml --output_zarr /datadrive/hadgem3/v8.2_monsoon_to_era5_40k_noise1 --checkpoint ~/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/checkpoints/gen_00040000.pt  --x2x b2a --seed 202109061100 --add_noise


python translate.py --config ~/model_outputs/outputs/v8.5_monsoon_hadgem_to_era5/config.yaml --output_zarr /datadrive/hadgem3/v8.5_monsoon_to_era5_90k --checkpoint ~/model_outputs/outputs/v8.5_monsoon_hadgem_to_era5/checkpoints/gen_00090000.pt  --x2x b2a --seed 202110032124