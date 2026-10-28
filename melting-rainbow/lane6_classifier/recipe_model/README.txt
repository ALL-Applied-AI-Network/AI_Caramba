RECIPE MODEL KIT - check Frank's EfficientNet honestly (Oct 26)
=====================================================================
Two scripts:
  score_video5_recipe.py   score the CURRENT model once on real video 5 (the check that matters)
  make_recipe_split.py     rebuild the twin train/val/test split without leaks or wrong labels

Run both on the DGX Spark, inside the same venv Frank trains in (it already has torch, timm, torchvision).

---------------------------------------------------------------------
0. PUT THE FILES ON THE SPARK (once)
---------------------------------------------------------------------
Copy these into ~/melting_rainbow on the Spark, the same way twin_dataset_v2 got there
(scp from your laptop, or upload in VS Code):
  recipe_model_kit.zip      (this kit)
  video5_eval.zip           (Drive: 01 Team Handoff > 03_validation_video5)
  twin_dataset_v2.zip       (Drive: 01 Team Handoff > 04_twin_datasets) - only for step 2
Then:
  cd ~/melting_rainbow
  source venv/bin/activate
  unzip -o recipe_model_kit.zip
  unzip -o video5_eval.zip

---------------------------------------------------------------------
1. SCORE THE CURRENT MODEL ON VIDEO 5 (about 1 minute)
---------------------------------------------------------------------
a) Find the saved weights:            ls *.pth *.pt
b) Check how evaluate.py prepares images (the score script must do the same):
     grep -n "Resize\|Normalize\|create_model\|efficientnet" train.py evaluate.py
   - Resize 224 + Normalize with 0.485/0.456/0.406 -> defaults are right, nothing to add
   - a different size, e.g. Resize((256,256))      -> add  --size 256
   - no Normalize line                              -> add  --no-normalize
   - torchvision model instead of timm (models.efficientnet_b0) -> tell Claude, one line changes
c) Run (replace model.pth with the real file name):
     python score_video5_recipe.py --weights model.pth --pack video5_eval
d) What you get (also saved as video5_recipe_scores.csv):
     - recipe accuracy on the 20 clean vials  -> does twin training carry over to the real camera?
     - misses / false alarms / cause right     -> same three numbers we report for Approach A
       (rule used: flag OFF-COLOR when predicted recipe != planned recipe; cause = predicted recipe)
     Approach A on the same 30 vials: 0 misses, 2 false alarms, cause 6/10.
e) Write the result into the experiment log as "Experiment 2 - video 5 check", with the model file name and date.
   Run this ONCE per model version. Don't change anything because of what you see in video 5 and re-score -
   that is tuning on the validation video (MAIC rule: train on the twin, validate on video 5).

---------------------------------------------------------------------
2. HONEST TWIN SPLIT (optional - read the warning first)
---------------------------------------------------------------------
  unzip -o twin_dataset_v2.zip
  python make_recipe_split.py --data twin_dataset_v2 --out recipe_split
Makes recipe_split/train, val, test (one recipe_<id> folder each, like ImageFolder wants) + split_manifest.csv.
Point train.py / evaluate.py at those three folders instead of splitting one folder at random.

WARNING - what this showed when we ran it (Oct 26): after removing leaked copies and vials that are mostly the
previous recipe, the 2,000-image set has only 63 different liquids with < 10% leftover (4-9 per recipe).
Test = 11 images. So a twin-only recipe accuracy can't say much either way - step 1 is the real test.
(--max-leftover 0.2 keeps 793 images / more renders if you want a bigger, slightly noisier set.)

Why the old split was misleading (our check of twin_dataset_v2/labels.csv, Oct 25):
  - 600 of the 1,315 "good" images are 60 augmented copies each of 10 clean renders; a random split puts copies of
    the same render in train AND test, so R5/R6/R7/R8/R14 scoring ~100% is mostly memorized copies.
  - 282 "good" images are >= 50% the previous recipe's liquid (e.g. 12 "R13" images are mostly R12, 10 "R12"
    images are mostly R13). They were labeled "good" because in the twin they looked within dE 5 of clean.
    That is where the R12 <-> R13 and R9 -> R13/R11 confusion comes from, not real carryover.

---------------------------------------------------------------------
3. WHERE THE MODEL SHOULD GO NEXT (talk as a team)
---------------------------------------------------------------------
The line already knows the recipe; the judges score misses / false alarms / cause, and want leave-one-recipe-out,
which a recipe classifier can't do (it can't name a recipe it never trained on). Better target for the AI part:
input = the vial crop + its batch's clean reference crop, output = good / off-color. It uses all 2,000 images
(including the 685 off-color ones), compares instead of memorizing absolute twin colors (twin vs real differ by
dE 16-52, our Oct 24 paper test), and can be tested leave-one-recipe-out. Claude can build the starter script.
