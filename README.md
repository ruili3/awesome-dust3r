<div align="center">
<h1>Awesome DUSt3R Resources </h1>
</div>

A curated list of papers and open-source resources related to DUSt3R/MASt3R, the emerging geometric foundation models empowering a wide span of 3D geometry tasks & applications. PR requests are welcomed, including papers, open-source libraries, blog posts, videos, etc. Repo maintained by [@Rui Li](https://x.com/leedaray), stay tuned for updates!

## Table of contents

- [Seminal Papers of DUSt3R](#seminal-papers-of-dust3r)
- [Concurrent Works](#concurrent-works)

<br>

- [Gaussian Splatting](#gaussian-splatting)
- [3D Reconstruction](#3d-reconstruction)
- [Robotics](#robotics)

<br>

- [Related Codebase](#related-codebase)
- [Blog Posts](#blog-posts)
- [Tutorial Videos](#tutorial-videos)
- [Acknowledgements](#acknowledgements)


<details span>
<summary><b>Update Log:</b></summary>

**Aug 31, 2024**: Add Spurfies.
<br>
**Aug 29, 2024**: Add Splatt3R, update the code of InstantSplat, etc.
<br>
**Jun 21, 2024**: Add the newly released MASt3R.
<br>
**May 31, 2024**: Add a concurrent work Detector-free SfM and a Mini-DUSt3R codebase.
<br>
**Apr 27, 2024**: Add concurrent works including FlowMap, ACE0, MicKey, and VGGSfM.
<br>
**Apr 09, 2024**: Initial list with first 3 papers, blogs and videos. 

</details>
<br>

## Seminal Papers of DUSt3R:
### 1. DUSt3R: Geometric 3D Vision Made Easy ![](https://img.shields.io/badge/2024-CVPR-green)
**Authors**: Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud

<details span>
<summary><b>Abstract</b></summary>
Multi-view stereo reconstruction (MVS) in the wild requires to first estimate the camera parameters e.g. intrinsic and extrinsic parameters. These are usually tedious and cumbersome to obtain, yet they are mandatory to triangulate corresponding pixels in 3D space, which is the core of all best performing MVS algorithms. In this work, we take an opposite stance and introduce DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses. We cast the pairwise reconstruction problem as a regression of pointmaps, relaxing the hard constraints of usual projective camera models. We show that this formulation smoothly unifies the monocular and binocular reconstruction cases. In the case where more than two images are provided, we further propose a simple yet effective global alignment strategy that expresses all pairwise pointmaps in a common reference frame. We base our network architecture on standard Transformer encoders and decoders, allowing us to leverage powerful pretrained models. Our formulation directly provides a 3D model of the scene as well as depth information, but interestingly, we can seamlessly recover from it, pixel matches, relative and absolute camera. Exhaustive experiments on all these tasks showcase that the proposed DUSt3R can unify various 3D vision tasks and set new SoTAs on monocular/multi-view depth estimation as well as relative pose estimation. In summary, DUSt3R makes many geometric 3D vision tasks easy.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2312.14132.pdf) | [🌐 Project Page](https://dust3r.europe.naverlabs.com/) | [⌨️ Code](https://github.com/naver/dust3r) | [🎥 Explanation Video](https://www.youtube.com/watch?v=JdfrG89iPOA) 

<br>


### 2. Grounding Image Matching in 3D with MASt3R ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Vincent Leroy, Yohann Cabon, Jérôme Revaud

<details span>
<summary><b>Abstract</b></summary>
Image Matching is a core component of all best-performing algorithms and pipelines in 3D vision. Yet despite matching being fundamentally a 3D problem, intrinsically linked to camera pose and scene geometry, it is typically treated as a 2D problem. This makes sense as the goal of matching is to establish correspondences between 2D pixel fields, but also seems like a potentially hazardous choice. In this work, we take a different stance and propose to cast matching as a 3D task with DUSt3R, a recent and powerful 3D reconstruction framework based on Transformers. Based on pointmaps regression, this method displayed impressive robustness in matching views with extreme viewpoint changes, yet with limited accuracy. We aim here to improve the matching capabilities of such an approach while preserving its robustness. We thus propose to augment the DUSt3R network with a new head that outputs dense local features, trained with an additional matching loss. We further address the issue of quadratic complexity of dense matching, which becomes prohibitively slow for downstream applications if not carefully treated. We introduce a fast reciprocal matching scheme that not only accelerates matching by orders of magnitude, but also comes with theoretical guarantees and, lastly, yields improved results. Extensive experiments show that our approach, coined MASt3R, significantly outperforms the state of the art on multiple matching tasks. In particular, it beats the best published methods by 30% (absolute improvement) in VCRE AUC on the extremely challenging Map-free localization dataset.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2406.09756) | [🌐 Project Page (to be released)](https://arxiv.org/pdf/2406.09756) | [⌨️ Code (to be released)](https://github.com/naver/mast3r)

<br>

### 3. CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion ![](https://img.shields.io/badge/2022-Neurips-blue)
**Authors**: Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Brégier, Yohann Cabon, Vaibhav Arora, Leonid Antsfeld, Boris Chidlovskii, Gabriela Csurka, Jérôme Revaud

<details span>
<summary><b>Abstract</b></summary>
Masked Image Modeling (MIM) has recently been established as a potent pre-training paradigm. A pretext task is constructed by masking patches in an input image, and this masked content is then predicted by a neural network using visible patches as sole input. This pre-training leads to state-of-the-art performance when finetuned for high-level semantic tasks, e.g. image classification and object detection. In this paper we instead seek to learn representations that transfer well to a wide variety of 3D vision and lower-level geometric downstream tasks, such as depth prediction or optical flow estimation. Inspired by MIM, we propose an unsupervised representation learning task trained from pairs of images showing the same scene from different viewpoints. More precisely, we propose the pretext task of cross-view completion where the first input image is partially masked, and this masked content has to be reconstructed from the visible content and the second image. In single-view MIM, the masked content often cannot be inferred precisely from the visible portion only, so the model learns to act as a prior influenced by high-level semantics. In contrast, this ambiguity can be resolved with cross-view completion from the second unmasked image, on the condition that the model is able to understand the spatial relationship between the two images. Our experiments show that our pretext task leads to significantly improved performance for monocular 3D vision downstream tasks such as depth estimation. In addition, our model can be directly applied to binocular downstream tasks like optical flow or relative camera pose estimation, for which we obtain competitive results without bells and whistles, i.e., using a generic architecture without any task-specific design.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2210.10716.pdf) | [🌐 Project Page](https://croco.europe.naverlabs.com/public/index.html) | [⌨️ Code](https://github.com/naver/croco)

<br>


### 4. CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow ![](https://img.shields.io/badge/2023-ICCV-f5cac3)
**Authors**: Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy, Yohann Cabon, Vaibhav Arora, Romain Brégier, Gabriela Csurka, Leonid Antsfeld, Boris Chidlovskii, Jérôme Revaud

<details span>
<summary><b>Abstract</b></summary>
Despite impressive performance for high-level downstream tasks, self-supervised pre-training methods have not yet fully delivered on dense geometric vision tasks such as stereo matching or optical flow. The application of self-supervised concepts, such as instance discrimination or masked image modeling, to geometric tasks is an active area of research. In this work, we build on the recent cross-view completion framework, a variation of masked image modeling that leverages a second view from the same scene which makes it well suited for binocular downstream tasks. The applicability of this concept has so far been limited in at least two ways: (a) by the difficulty of collecting real-world image pairs -- in practice only synthetic data have been used -- and (b) by the lack of generalization of vanilla transformers to dense downstream tasks for which relative position is more meaningful than absolute position. We explore three avenues of improvement. First, we introduce a method to collect suitable real-world image pairs at large scale. Second, we experiment with relative positional embeddings and show that they enable vision transformers to perform substantially better. Third, we scale up vision transformer based cross-completion architectures, which is made possible by the use of large amounts of data. With these improvements, we show for the first time that state-of-the-art results on stereo matching and optical flow can be reached without using any classical task-specific techniques like correlation volume, iterative estimation, image warping or multi-scale reasoning, thus paving the way towards universal vision models.
</details>
  
 [📃 Paper](https://arxiv.org/abs/2211.10408) | [🌐 Project Page](https://croco.europe.naverlabs.com/public/index.html) | [⌨️ Code](https://github.com/naver/croco)

<br>




## Concurrent Works:
## 2024:
### 1. FlowMap: High-Quality Camera Poses, Intrinsics, and Depth via Gradient Descent ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Cameron Smith, David Charatan, Ayush Tewari, Vincent Sitzmann

<details span>
<summary><b>Abstract</b></summary>
This paper introduces FlowMap, an end-to-end differentiable method that solves for precise camera poses, camera intrinsics, and per-frame dense depth of a video sequence. Our method performs per-video gradient-descent minimization of a simple least-squares objective that compares the optical flow induced by depth, intrinsics, and poses against correspondences obtained via off-the-shelf optical flow and point tracking. Alongside the use of point tracks to encourage long-term geometric consistency, we introduce a differentiable re-parameterization of depth, intrinsics, and pose that is amenable to first-order optimization. We empirically show that camera parameters and dense depth recovered by our method enable photo-realistic novel view synthesis on 360° trajectories using Gaussian Splatting. Our method not only far outperforms prior gradient-descent based bundle adjustment methods, but surprisingly performs on par with COLMAP, the state-of-the-art SfM method, on the downstream task of 360° novel view synthesis - even though our method is purely gradient-descent based, fully differentiable, and presents a complete departure from conventional SfM. Our result opens the door to the self-supervised training of neural networks that perform camera parameter estimation, 3D reconstruction, and novel view synthesis.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2404.15259) | [🌐 Project Page](https://cameronosmith.github.io/flowmap/) | [⌨️ Code](https://github.com/dcharatan/flowmap)

<br>

### 2. Scene Coordinate Reconstruction: Posing of Image Collections via Incremental Learning of a Relocalizer ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Eric Brachmann, Jamie Wynn, Shuai Chen, Tommaso Cavallari, Áron Monszpart, Daniyar Turmukhambetov, Victor Adrian Prisacariu

<details span>
<summary><b>Abstract</b></summary>
We address the task of estimating camera parameters from a set of images depicting a scene. Popular feature-based structure-from-motion (SfM) tools solve this task by incremental reconstruction: they repeat triangulation of sparse 3D points and registration of more camera views to the sparse point cloud. We re-interpret incremental structure-from-motion as an iterated application and refinement of a visual relocalizer, that is, of a method that registers new views to the current state of the reconstruction. This perspective allows us to investigate alternative visual relocalizers that are not rooted in local feature matching. We show that scene coordinate regression, a learning-based relocalization approach, allows us to build implicit, neural scene representations from unposed images. Different from other learning-based reconstruction methods, we do not require pose priors nor sequential inputs, and we optimize efficiently over thousands of images. Our method, ACE0 (ACE Zero), estimates camera poses to an accuracy comparable to feature-based SfM, as demonstrated by novel view synthesis.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2404.14351) | [🌐 Project Page](https://nianticlabs.github.io/acezero/) | [⌨️ Code](https://github.com/nianticlabs/acezero)

<br>

### 3. Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences ![](https://img.shields.io/badge/2024-CVPR-green)
**Authors**: Axel Barroso-Laguna, Sowmya Munukutla, Victor Adrian Prisacariu, Eric Brachmann

<details span>
<summary><b>Abstract</b></summary>
Given two images, we can estimate the relative camera pose between them by establishing image-to-image correspondences. Usually, correspondences are 2D-to-2D and the pose we estimate is defined only up to scale. Some applications, aiming at instant augmented reality anywhere, require scale-metric pose estimates, and hence, they rely on external depth estimators to recover the scale. We present MicKey, a keypoint matching pipeline that is able to predict metric correspondences in 3D camera space. By learning to match 3D coordinates across images, we are able to infer the metric relative pose without depth measurements. Depth measurements are also not required for training, nor are scene reconstructions or image overlap information. MicKey is supervised only by pairs of images and their relative poses. MicKey achieves state-of-the-art performance on the Map-Free Relocalisation benchmark while requiring less supervision than competing approaches.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2404.06337) | [🌐 Project Page](https://nianticlabs.github.io/mickey/) | [⌨️ Code](https://github.com/nianticlabs/mickey)

<br>

### 4. VGGSfM: Visual Geometry Grounded Deep Structure From Motion ![](https://img.shields.io/badge/2024-CVPR-green)
**Authors**: Jianyuan Wang, Nikita Karaev, Christian Rupprecht, David Novotny

<details span>
<summary><b>Abstract</b></summary>
Structure-from-motion (SfM) is a long-standing problem in the computer vision community, which aims to reconstruct the camera poses and 3D structure of a scene from a set of unconstrained 2D images. Classical frameworks solve this problem in an incremental manner by detecting and matching keypoints, registering images, triangulating 3D points, and conducting bundle adjustment. Recent research efforts have predominantly revolved around harnessing the power of deep learning techniques to enhance specific elements (e.g., keypoint matching), but are still based on the original, non-differentiable pipeline. Instead, we propose a new deep SfM pipeline VGGSfM, where each component is fully differentiable and thus can be trained in an end-to-end manner. To this end, we introduce new mechanisms and simplifications. First, we build on recent advances in deep 2D point tracking to extract reliable pixel-accurate tracks, which eliminates the need for chaining pairwise matches. Furthermore, we recover all cameras simultaneously based on the image and track features instead of gradually registering cameras. Finally, we optimise the cameras and triangulate 3D points via a differentiable bundle adjustment layer. We attain state-of-the-art performance on three popular datasets, CO3D, IMC Phototourism, and ETH3D.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2312.04563) | [🌐 Project Page](https://vggsfm.github.io/) | [⌨️ Code](https://github.com/facebookresearch/vggsfm)

<br>

### 5. Detector-Free Structure from Motion ![](https://img.shields.io/badge/2024-CVPR-green)
**Authors**: Xingyi He, Jiaming Sun, Yifan Wang, Sida Peng, Qixing Huang, Hujun Bao, Xiaowei Zhou

<details span>
<summary><b>Abstract</b></summary>
We propose a new structure-from-motion framework to recover accurate camera poses and point clouds from unordered images. Traditional SfM systems typically rely on the successful detection of repeatable keypoints across multiple views as the first step, which is difficult for texture-poor scenes, and poor keypoint detection may break down the whole SfM system. We propose a new detector-free SfM framework to draw benefits from the recent success of detector-free matchers to avoid the early determination of keypoints, while solving the multi-view inconsistency issue of detector-free matchers. Specifically, our framework first reconstructs a coarse SfM model from quantized detector-free matches. Then, it refines the model by a novel iterative refinement pipeline, which iterates between an attention-based multi-view matching module to refine feature tracks and a geometry refinement module to improve the reconstruction accuracy. Experiments demonstrate that the proposed framework outperforms existing detector-based SfM systems on common benchmark datasets. We also collect a texture-poor SfM dataset to demonstrate the capability of our framework to reconstruct texture-poor scenes. Based on this framework, we take first place in Image Matching Challenge 2023.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2306.15669) | [🌐 Project Page](https://zju3dv.github.io/DetectorFreeSfM/) | [⌨️ Code](https://github.com/zju3dv/DetectorFreeSfM)

<br>




## Gaussian Splatting:
## 2024:
### 1. InstantSplat: Unbounded Sparse-view Pose-free Gaussian Splatting in 40 Seconds ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, Zhangyang Wang, Yue Wang
<details span>
<summary><b>Abstract</b></summary>
While novel view synthesis (NVS) has made substantial progress in 3D computer vision, it typically requires an initial estimation of camera intrinsics and extrinsics from dense viewpoints. This pre-processing is usually conducted via a Structure-from-Motion (SfM) pipeline, a procedure that can be slow and unreliable, particularly in sparse-view scenarios with insufficient matched features for accurate reconstruction. In this work, we integrate the strengths of point-based representations (e.g., 3D Gaussian Splatting, 3D-GS) with end-to-end dense stereo models (DUSt3R) to tackle the complex yet unresolved issues in NVS under unconstrained settings, which encompasses pose-free and sparse view challenges. Our framework, InstantSplat, unifies dense stereo priors with 3D-GS to build 3D Gaussians of large-scale scenes from sparseview & pose-free images in less than 1 minute. Specifically, InstantSplat comprises a Coarse Geometric Initialization (CGI) module that swiftly establishes a preliminary scene structure and camera parameters across all training views, utilizing globally-aligned 3D point maps derived from a pre-trained dense stereo pipeline. This is followed by the Fast 3D-Gaussian Optimization (F-3DGO) module, which jointly optimizes the 3D Gaussian attributes and the initialized poses with pose regularization. Experiments conducted on the large-scale outdoor Tanks & Temples datasets demonstrate that InstantSplat significantly improves SSIM (by 32%) while concurrently reducing Absolute Trajectory Error (ATE) by 80%. These establish InstantSplat as a viable solution for scenarios involving posefree and sparse-view conditions. Project page: http://instantsplat.github.io/.
</details>

  [📄 Paper](https://arxiv.org/pdf/2403.20309.pdf) | [🌐 Project Page](https://instantsplat.github.io/) | [💻 Code](https://github.com/NVlabs/InstantSplat) | [🎥 Video](https://www.youtube.com/watch?v=_9aQHLHHoEM&feature=youtu.be) 

<br>


### 2. Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Brandon Smart, Chuanxia Zheng, Iro Laina, Victor Adrian Prisacariu
<details span>
<summary><b>Abstract</b></summary>
In this paper, we introduce Splatt3R, a pose-free, feed-forward method for in-the-wild 3D reconstruction and novel view synthesis from stereo pairs. Given uncalibrated natural images, Splatt3R can predict 3D Gaussian Splats without requiring any camera parameters or depth information. For generalizability, we build Splatt3R upon a ``foundation'' 3D geometry reconstruction method, MASt3R, by extending it to deal with both 3D structure and appearance. Specifically, unlike the original MASt3R which reconstructs only 3D point clouds, we predict the additional Gaussian attributes required to construct a Gaussian primitive for each point. Hence, unlike other novel view synthesis methods, Splatt3R is first trained by optimizing the 3D point cloud's geometry loss, and then a novel view synthesis objective. By doing this, we avoid the local minima present in training 3D Gaussian Splats from stereo views. We also propose a novel loss masking strategy that we empirically find is critical for strong performance on extrapolated viewpoints. We train Splatt3R on the ScanNet++ dataset and demonstrate excellent generalisation to uncalibrated, in-the-wild images. Splatt3R can reconstruct scenes at 4FPS at 512 x 512 resolution, and the resultant splats can be rendered in real-time.
</details>

  [📄 Paper](https://arxiv.org/pdf/2408.13912) | [🌐 Project Page](https://splatt3r.active.vision/) | [💻 Code](https://github.com/btsmart/splatt3r)

<br>




## 3D Reconstruction:
## 2024:
### 1. Spurfies: Sparse Surface Reconstruction using Local Geometry Priors ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Kevin Raj, Christopher Wewer, Raza Yunus, Eddy Ilg, Jan Eric Lenssen
<details span>
<summary><b>Abstract</b></summary>
We introduce Spurfies, a novel method for sparse-view surface reconstruction that disentangles appearance and geometry information to utilize local geometry priors trained on synthetic data. Recent research heavily focuses on 3D reconstruction using dense multi-view setups, typically requiring hundreds of images. However, these methods often struggle with few-view scenarios. Existing sparse-view reconstruction techniques often rely on multi-view stereo networks that need to learn joint priors for geometry and appearance from a large amount of data. In contrast, we introduce a neural point representation that disentangles geometry and appearance to train a local geometry prior using a subset of the synthetic ShapeNet dataset only. During inference, we utilize this surface prior as additional constraint for surface and appearance reconstruction from sparse input views via differentiable volume rendering, restricting the space of possible solutions. We validate the effectiveness of our method on the DTU dataset and demonstrate that it outperforms previous state of the art by 35% in surface quality while achieving competitive novel view synthesis quality. Moreover, in contrast to previous works, our method can be applied to larger, unbounded scenes, such as Mip-NeRF 360.
</details>

  [📄 Paper](https://arxiv.org/pdf/2408.16544) | [🌐 Project Page](https://geometric-rl.mpi-inf.mpg.de/spurfies/index.html) | [💻 Code (coming soom)](https://geometric-rl.mpi-inf.mpg.de/spurfies/index.html)

<br>

## Robotics:
## 2024:
### 1. Unifying Scene Representation and Hand-Eye Calibration with 3D Foundation Models ![](https://img.shields.io/badge/2024-RAL-yellow)
**Authors**: Weiming Zhi, Haozhan Tang, Tianyi Zhang, Matthew Johnson-Roberson
<details span>
<summary><b>Abstract</b></summary>
Representing the environment is a central challenge in robotics, and is essential for effective decision-making. Traditionally, before capturing images with a manipulatormounted camera, users need to calibrate the camera using a specific external marker, such as a checkerboard or AprilTag.
However, recent advances in computer vision have led to the development of 3D foundation models. These are large, pre-trained neural networks that can establish fast and accurate multi-view correspondences with very few images, even in the absence of rich visual features. This paper advocates for the integration of 3D foundation models into scene representation approaches for robotic systems equipped with manipulator-mounted RGB cameras. Specifically, we propose the Joint Calibration and Representation (JCR) method. JCR uses RGB images, captured by a manipulator-mounted camera, to simultaneously construct an environmental representation and calibrate the camera relative to the robot’s end-effector, in the absence of specific calibration markers. The resulting 3D environment representation is aligned with the robot’s coordinate frame and maintains physically accurate scales. We demonstrate that JCR can build effective scene representations using a low-cost RGB camera attached to a manipulator, without prior calibration.
</details>

  [📄 Paper](https://arxiv.org/pdf/2404.11683.pdf) | [💻 Code (to be released)]()

<br>

### 2. 3D Foundation Models Enable Simultaneous Geometry and Pose Estimation of Grasped Objects ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Weiming Zhi, Haozhan Tang, Tianyi Zhang, Matthew Johnson-Roberson
<details span>
<summary><b>Abstract</b></summary>
Humans have the remarkable ability to use held objects as tools to interact with their environment. For this to occur, humans internally estimate how hand movements affect the object’s movement. We wish to endow robots with this capability. We contribute methodology to jointly estimate the geometry and pose of objects grasped by a robot, from RGB images captured by an external camera. Notably, our method transforms the estimated geometry into the robot’s coordinate frame, while not requiring the extrinsic parameters of the external camera to be calibrated. Our approach leverages 3D foundation models, large models pre-trained on huge datasets for 3D vision tasks, to produce initial estimates of the in-hand object. These initial estimations do not have physically correct scales and are in the camera’s frame. Then, we formulate, and efficiently solve, a coordinate-alignment problem to recover accurate scales, along with a transformation of the objects to the coordinate frame of the robot. Forward kinematics mappings can subsequently be defined from the manipulator’s joint angles to specified points on the object. These mappings enable the estimation of points on the held object at arbitrary configurations, enabling robot motion to be designed with respect to coordinates on the grasped objects. We empirically evaluate our approach on a robot manipulator holding a diverse set of real-world objects.
</details>

[📄 Paper](https://www.researchgate.net/profile/Weiming-Zhi/publication/382490016_3D_Foundation_Models_Enable_Simultaneous_Geometry_and_Pose_Estimation_of_Grasped_Objects/links/66a01a4527b00e0ca43ddd95/3D-Foundation-Models-Enable-Simultaneous-Geometry-and-Pose-Estimation-of-Grasped-Objects.pdf)
<br>

## Related Codebase
1. [Mini-DUSt3R](https://github.com/pablovela5620/mini-dust3r): A miniature version of dust3r only for performing inference. May, 2024.

## Blog Posts

1. [3D reconstruction models made easy](https://europe.naverlabs.com/blog/3d-reconstruction-models-made-easy/)
2. [InstantSplat: Sub Minute Gaussian Splatting](https://radiancefields.com/instantsplat-sub-minute-gaussian-splatting/)


## Tutorial Videos
1. [Advanced Image-to-3D AI, DUSt3R](https://www.youtube.com/watch?v=kI7wCEAFFb0)
2. [BSLIVE Pinokio Dust3R to turn 2D into 3D Mesh](https://www.youtube.com/watch?v=vY7GcbOsC-U)
3. [InstantSplat, DUSt3R](https://www.youtube.com/watch?v=JdfrG89iPOA)

## Acknowledgements
- Thanks to [Janusch](https://twitter.com/janusch_patas) for the awesome paper list [awesome-3D-gaussian-splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting) and to [Chao Wen](https://walsvid.github.io/) for the [Awesome-MVS](https://github.com/walsvid/Awesome-MVS). This list was designed with reference to both.
