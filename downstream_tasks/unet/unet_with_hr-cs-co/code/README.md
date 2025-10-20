# FineTune Sysmifta using SSL features

Before executing the processing chain, edit the configuration file sysmifta.cfg with the correct settings.

Optional, all of the functions below can take a configuration filename as an input by specifying '-c <filename>', if omitted the default filename (sysmifta.cfg) is used.

Processing chain:
1. python3 extract_masks_from_cytomine.py (will ask for the user's Public Key and Private Key from their cytomine account, instructions will be printed during execution)
2. python3 background_tissue_detection.py
3. python3 create_ground_truths.py
4. python3 extract_patches.py
5. python3 augment_patches.py (if not using live augmentation)
6. python3 downsample_patches.py
7. python3 train_unet.py (will output the label of the saved network)
8. python3 train_threshold.py <label from step 6> (does not work in Python 3)
9. python3 apply_unet.py <label from step 6> (segments the test images)
   or
   python3 test_unet.py <label from step 6> (test the network on the test patches)
10. python3 pixel_based_evaluation.py <label from step 6>


Limitations:
 - Only one class can exist per pixel (currently the last item in the [classnumbers] section of the config file that exists in a location will be the label)