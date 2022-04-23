# gwsol
The pytorch code of "Generalized Weakly Supervised Object Localization". The code was based on code <https://github.com/ZJULearning/AttentionZSL> and code <https://github.com/junsukchoe/ADL> . Thanks for their nice job!
![微信截图_20220319193914](https://user-images.githubusercontent.com/83970726/159119629-38f7888a-c269-4209-bc32-e2aba5d85dca.png)

# Installation
The project needs 1 NVIDIA 1080TI, python=3.6  
pytorch=1.0.1  
opencv-python=3.4.3.18  
matplotlib=3.2.1  
numpy=1.16.1  
Pillow=6.1.0  
PyYAML=5.3  
scikit-learn=0.22.2.post1  

# Data preparation

Experiments are conducted on AwA2 <https://cvml.ist.ac.at/AwA2/>, CUB <http://www.vision.caltech.edu/visipedia/CUB-200.html> datasets.
When you download the dataset, you put the data into "/your_home_root/gwsol/data/".  
<pre><code>--AwA2  
<pre><code>--JPEGImages  
--proposed_split  
--classes.txt  
--predicate-matrix-continuous.txt  
...  
</code></pre>
</code></pre>
<pre><code>--CUB  
<pre><code>--JPEGImages  
--proposed_split  
--classes.txt  
--predicate-matrix-continuous.txt  
...  
</code></pre>
</code></pre>
# Train:

CUB:  
<pre><code>
python experiments/run_trainer.py --cfg ./configs/hybrid/VGG19_CUB_PS_C.yaml
</code></pre>
AWA2:  
<pre><code>
python experiments/run_trainer.py --cfg ./configs/hybrid/VGG19_AwA2_PS_C.yaml
</code></pre>

# Test:
Before you test your model, you should change the "ckpt_name" in the "/Your_Home_Root/gwsol/configs/hybird/VGG19_CUB_PS_C.yaml", such as "VGG19_CUB_PS_C_2021-03-02-13-46"  

## CUB:   
C setting:  
<pre><code>
python experiments/run_evaluator_hybrid.py --cfg ./configs/hybrid/VGG19_CUB_PS_C.yaml    
</code></pre>

G setting:  
<pre><code>
python experiments/run_evaluator_hybrid.py --cfg ./configs/hybrid/VGG19_CUB_PS_G.yaml   
</code></pre>

## AWA2:   
C setting:  
<pre><code>
python experiments/run_evaluator_hybrid.py --cfg ./configs/hybrid/VGG19_AwA2_PS_C.yaml   
</code></pre>

G setting:  
<pre><code>
python experiments/run_evaluator_hybrid.py --cfg ./configs/hybrid/VGG19_AwA2_PS_G.yaml   
</code></pre>


#  AwA2 Boxes Annotation 
In order to get metrics on the AwA2 dataset, we manually labeled the test dataset of AwA2, you can download the annotation from the dropbox <https://www.dropbox.com/scl/fo/jbzry4jrad1800rkr71nb/h?dl=0&rlkey=wdnz6ptsedfl9umgpjvreolv9>, Please put them into the folder "You_Home_Root/gwsol/loc_evaluation/awa2/", like "You_Home_Root/gwsol/loc_evaluation/awa2/test_seen_gt/"

# Contact US  
If you have some questions about this project, please contact me, my email is wyzeng2019@gmail.com

