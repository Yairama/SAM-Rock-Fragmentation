# SAM Rock Fragmentation


<p>
  <a href="https://www.linkedin.com/in/yairama/" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin" class="icon" width="20" height="20"> LinkedIn
  </a> &nbsp; 
  <a href="https://github.com/Yairama" rel="nofollow noreferrer">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="github" class="icon" width="20" height="20"> Github
  </a> &nbsp; 
  <a href="https://gitlab.com/Yairama" rel="nofollow noreferrer">
    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968853.png" alt="gitlab" class="icon" width="20" height="20"> Gitlab
  </a>
</p>

The goal of the SAM Rock Fragmentation repository is to use the recent [Segment Anything](https://segment-anything.com) mask generator model created by [MetaAI](https://ai.facebook.com/research/) and implement it with an interface and set of algorithms to calculate the P80 resulting from a blasting.

*It is not recommended to use the project in production (not yet) because the SAM model was designed for the detection of objects and not rock fragments as such. However, it is useful to provide a proof of concept for future better projects.*


## About Segment Anything

[The Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

SAM in rocks fragments looks like this:
![image](https://user-images.githubusercontent.com/45445692/232190344-8b8a24d6-e033-4f4e-b011-d143510adfa1.png)

## About the tool
- The tool was developed using the Streamit library in Python.
- The images used as examples in this repository were extracted from https://data.mendeley.com/datasets/z78ghz96bn/1.
- The model used for the inference is vit_b because it requires less resources, however, with small changes in the code you can use any model you want. for more information see [Models Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
- The diameter of the red balls of the images is used as a scale reference, therefore it is necessary that the image you are going to use has at least one red ball.
- Currently the tool does not take into consideration the depth of the image, this implementation is likely to be implemented in future versions.

## How to use

The notebook *demo.ipynb* has been implemented where the tool can be used through the Google Colab platform, just open this **[link](https://colab.research.google.com/github/Yairama/SAM-Rock-Fragmentation/blob/main/demo.ipynb)**, **Change runtime type to GPU** and execute all the cells, finally in the last cell a link will be generated where the tool can be used. Please be patient with the loading times, being a free platform in the cloud it may take some time to run.

https://user-images.githubusercontent.com/45445692/232190032-27ef4c73-e323-4532-8355-a51cd5de62c8.mp4


## How can you use your own images? 
If you want to use your own images just upload them into the images folder generated in Google Colab and reload the page (see the capture).

![image](https://user-images.githubusercontent.com/45445692/232187536-b9f2fdd3-4ba0-4743-8c9c-4ca0f6d95064.png)

## Opportunities for improvement
- Results can be improved by using the SAM vit_h type model which has better results.
- If you have the resources, you can try Fine Tuning the SAM model with images of rock fragments, to increase the quality of the inference in the images.
- It can be combined with other models that detect the depth of the image in order to have better measurements on rock fragments.
- Whatever comes to your mind...

## Abour the results :)
The results are generally good, however in some images they can be complicated to segment for example, the image 41 is a difficult image to segment by SAM due the rocks dimensions:
![image](https://user-images.githubusercontent.com/45445692/232190228-286f3237-b18f-44e0-ba47-20f928d560b1.png)



## Some images of the tool:

Image from Dataset:
![image](https://user-images.githubusercontent.com/45445692/232190127-f5977284-877a-4662-9984-6ba25702800c.png)

Masked Imagen:
![image](https://user-images.githubusercontent.com/45445692/232190137-4c7efd21-d496-47e5-b5ba-946c607bb765.png)

P80 Area:
![image](https://user-images.githubusercontent.com/45445692/232190169-0d0707b4-319d-45eb-8c48-601314b61366.png)

P80 Diameter:
![image](https://user-images.githubusercontent.com/45445692/232190174-39370545-8c96-476e-a854-06e1038021b9.png)


## I hope you find it useful
anything to the dm :)

![image](https://media.tenor.com/mMVnCaqJ4D8AAAAM/loli-dance.gif)
