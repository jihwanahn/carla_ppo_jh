import splitfolders


output = "./out_split" #Enter Output Folder

splitfolders.ratio(input="_out", output=output, seed=42, ratio=(0.8,0.0,0.2))