<img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/dafidofff/9252e96ccc8465bbcf31f9c1ed1fbcbc/raw/covbadge.json" />

# Seg And Diffuse Repo 

This repo is based on the following Medium article: https://medium.com/@amir_shakiba/sam-grounding-dino-stable-diffusion-segment-detect-change-da7926947286

To properly install the Grounded_Segment_Anything repo we need to install all requirements. This can be done with the following steps: 
!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
%cd /content/Grounded-Segment-Anything 
!pip install -q -r requirements.txt or with poetry cat requirements.txt | xargs poetry add  
%cd /content/Grounded-Segment-Anything/GroundingDINO
!pip install -q -r requirements.txt or with poetry cat requirements.txt | xargs poetry add  
%cd /content/Grounded-Segment-Anything/segment_anything
!pip install -q -r requirements.txt or with poetry cat requirements.txt | xargs poetry add  
%cd /content/Grounded-Segment-Anything
