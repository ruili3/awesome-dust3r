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
- [Dynamic Scene Reconstruction](#dynamic-scene-reconstruction)
- [Scene Understanding](#scene-understanding)
- [Robotics](#robotics)

<br>

- [Related Codebase](#related-codebase)
- [Blog Posts](#blog-posts)
- [Tutorial Videos](#tutorial-videos)
- [Acknowledgements](#acknowledgements)


<details span>
<summary><b>Update Log:</b></summary>

**Dec 20, 2024**: Add Align3R.
<br>
**Nov 15, 2024**: Add MoGe, LSM.
<br>
**Oct 10, 2024**: Add MASt3R-SfM, MonST3R.
<br>
**Aug 31, 2024**: Add Spurfies, Spann3R, and ReconX.
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
  
 [📃 Paper](https://arxiv.org/pdf/2406.09756) | [🌐 Project Page (to be released)](https://arxiv.org/pdf/2406.09756) | [⌨️ Code](https://github.com/naver/mast3r)

<br>


### 3. MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, Jerome Revaud

<details span>
<summary><b>Abstract</b></summary>
Structure-from-Motion (SfM), a task aiming at jointly recovering camera poses and 3D geometry of a scene given a set of images, remains a hard problem with still many open challenges despite decades of significant progress. The traditional solution for SfM consists of a complex pipeline of minimal solvers which tends to propagate errors and fails when images do not sufficiently overlap, have too little motion, etc. Recent methods have attempted to revisit this paradigm, but we empirically show that they fall short of fixing these core issues. In this paper, we propose instead to build upon a recently released foundation model for 3D vision that can robustly produce local 3D reconstructions and accurate matches. We introduce a low-memory approach to accurately align these local reconstructions in a global coordinate system. We further show that such foundation models can serve as efficient image retrievers without any overhead, reducing the overall complexity from quadratic to linear. Overall, our novel SfM pipeline is simple, scalable, fast and truly unconstrained, i.e. it can handle any collection of images, ordered or not. Extensive experiments on multiple benchmarks show that our method provides steady performance across diverse settings, especially outperforming existing methods in small- and medium-scale settings.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2409.19152) | [🌐 Project Page](https://github.com/naver/mast3r) | [⌨️ Code](https://github.com/naver/mast3r)

<br>

### 4. CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion ![](https://img.shields.io/badge/2022-Neurips-blue)
**Authors**: Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Brégier, Yohann Cabon, Vaibhav Arora, Leonid Antsfeld, Boris Chidlovskii, Gabriela Csurka, Jérôme Revaud

<details span>
<summary><b>Abstract</b></summary>
Masked Image Modeling (MIM) has recently been established as a potent pre-training paradigm. A pretext task is constructed by masking patches in an input image, and this masked content is then predicted by a neural network using visible patches as sole input. This pre-training leads to state-of-the-art performance when finetuned for high-level semantic tasks, e.g. image classification and object detection. In this paper we instead seek to learn representations that transfer well to a wide variety of 3D vision and lower-level geometric downstream tasks, such as depth prediction or optical flow estimation. Inspired by MIM, we propose an unsupervised representation learning task trained from pairs of images showing the same scene from different viewpoints. More precisely, we propose the pretext task of cross-view completion where the first input image is partially masked, and this masked content has to be reconstructed from the visible content and the second image. In single-view MIM, the masked content often cannot be inferred precisely from the visible portion only, so the model learns to act as a prior influenced by high-level semantics. In contrast, this ambiguity can be resolved with cross-view completion from the second unmasked image, on the condition that the model is able to understand the spatial relationship between the two images. Our experiments show that our pretext task leads to significantly improved performance for monocular 3D vision downstream tasks such as depth estimation. In addition, our model can be directly applied to binocular downstream tasks like optical flow or relative camera pose estimation, for which we obtain competitive results without bells and whistles, i.e., using a generic architecture without any task-specific design.
</details>
  
 [📃 Paper](https://arxiv.org/pdf/2210.10716.pdf) | [🌐 Project Page](https://croco.europe.naverlabs.com/public/index.html) | [⌨️ Code](https://github.com/naver/croco)

<br>


### 5. CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow ![](https://img.shields.io/badge/2023-ICCV-f5cac3)
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




### 3. Dense Point Clouds Matter: Dust-GS for Scene Reconstruction from Sparse Viewpoints ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Shan Chen, Jiale Zhou, Lei Li
<details span>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in scene synthesis and novel view synthesis tasks. Typically, the initialization of 3D Gaussian primitives relies on point clouds derived from Structure-from-Motion (SfM) methods. However, in scenarios requiring scene reconstruction from sparse viewpoints, the effectiveness of 3DGS is significantly constrained by the quality of these initial point clouds and the limited number of input images. In this study, we present Dust-GS, a novel framework specifically designed to overcome the limitations of 3DGS in sparse viewpoint conditions. Instead of relying solely on SfM, Dust-GS introduces an innovative point cloud initialization technique that remains effective even with sparse input data. Our approach leverages a hybrid strategy that integrates an adaptive depth-based masking technique, thereby enhancing the accuracy and detail of reconstructed scenes. Extensive experiments conducted on several benchmark datasets demonstrate that Dust-GS surpasses traditional 3DGS methods in scenarios with sparse viewpoints, achieving superior scene reconstruction quality with a reduced number of input images.
</details>

  [📄 Paper](https://arxiv.org/pdf/2409.08613)

<br>



### 4. LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Hanyang Yu, Xiaoxiao Long, Ping Tan
<details span>
<summary><b>Abstract</b></summary>
We aim to address sparse-view reconstruction of a 3D scene by leveraging priors from large-scale vision models. While recent advancements such as 3D Gaussian Splatting (3DGS) have demonstrated remarkable successes in 3D reconstruction, these methods typically necessitate hundreds of input images that densely capture the underlying scene, making them time-consuming and impractical for real-world applications. However, sparse-view reconstruction is inherently ill-posed and under-constrained, often resulting in inferior and incomplete outcomes. This is due to issues such as failed initialization, overfitting on input images, and a lack of details. To mitigate these challenges, we introduce LM-Gaussian, a method capable of generating high-quality reconstructions from a limited number of images. Specifically, we propose a robust initialization module that leverages stereo priors to aid in the recovery of camera poses and the reliable point clouds. Additionally, a diffusion-based refinement is iteratively applied to incorporate image diffusion priors into the Gaussian optimization process to preserve intricate scene details. Finally, we utilize video diffusion priors to further enhance the rendered images for realistic visual effects. Overall, our approach significantly reduces the data acquisition requirements compared to previous 3DGS methods. We validate the effectiveness of our framework through experiments on various public datasets, demonstrating its potential for high-quality 360-degree scene reconstruction. Visual results are on our website.
</details>

  [📄 Paper](https://arxiv.org/pdf/2409.03456) | [🌐 Project Page](https://hanyangyu1021.github.io/lm-gaussian.github.io/) | [💻 Code](https://github.com/hanyangyu1021/LMGaussian)

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


### 2. 3D Reconstruction with Spatial Memory ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Hengyi Wang, Lourdes Agapito
<details span>
<summary><b>Abstract</b></summary>
We present Spann3R, a novel approach for dense 3D reconstruction from ordered or unordered image collections. Built on the DUSt3R paradigm, Spann3R uses a transformer-based architecture to directly regress pointmaps from images without any prior knowledge of the scene or camera parameters. Unlike DUSt3R, which predicts per image-pair pointmaps each expressed in its local coordinate frame, Spann3R can predict per-image pointmaps expressed in a global coordinate system, thus eliminating the need for optimization-based global alignment. The key idea of Spann3R is to manage an external spatial memory that learns to keep track of all previous relevant 3D information. Spann3R then queries this spatial memory to predict the 3D structure of the next frame in a global coordinate system. Taking advantage of DUSt3R's pre-trained weights, and further fine-tuning on a subset of datasets, Spann3R shows competitive performance and generalization ability on various unseen datasets and can process ordered image collections in real time.
</details>

  [📄 Paper](https://arxiv.org/pdf/2408.16061) | [🌐 Project Page](https://hengyiwang.github.io/projects/spanner) | [💻 Code (coming soom)](https://hengyiwang.github.io/projects/spanner)

<br>

### 3. ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, Yueqi Duan
<details span>
<summary><b>Abstract</b></summary>
Advancements in 3D scene reconstruction have transformed 2D images from the real world into 3D models, producing realistic 3D results from hundreds of input photos. Despite great success in dense-view reconstruction scenarios, rendering a detailed scene from insufficient captured views is still an ill-posed optimization problem, often resulting in artifacts and distortions in unseen areas. In this paper, we propose ReconX, a novel 3D scene reconstruction paradigm that reframes the ambiguous reconstruction challenge as a temporal generation task. The key insight is to unleash the strong generative prior of large pre-trained video diffusion models for sparse-view reconstruction. However, 3D view consistency struggles to be accurately preserved in directly generated video frames from pre-trained models. To address this, given limited input views, the proposed ReconX first constructs a global point cloud and encodes it into a contextual space as the 3D structure condition. Guided by the condition, the video diffusion model then synthesizes video frames that are both detail-preserved and exhibit a high degree of 3D consistency, ensuring the coherence of the scene from various perspectives. Finally, we recover the 3D scene from the generated video through a confidence-aware 3D Gaussian Splatting optimization scheme. Extensive experiments on various real-world datasets show the superiority of our ReconX over state-of-the-art methods in terms of quality and generalizability.
</details>

  [📄 Paper](https://arxiv.org/pdf/2408.16767) | [🌐 Project Page](https://liuff19.github.io/ReconX/) | [💻 Code](https://github.com/liuff19/ReconX)

<br>




### 4. MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, Jiaolong Yang
<details span>
<summary><b>Abstract</b></summary>
We present MoGe, a powerful model for recovering 3D geometry from monocular open-domain images. Given a single image, our model directly predicts a 3D point map of the captured scene with an affine-invariant representation, which is agnostic to true global scale and shift. This new representation precludes ambiguous supervision in training and facilitate effective geometry learning. Furthermore, we propose a set of novel global and local geometry supervisions that empower the model to learn high-quality geometry. These include a robust, optimal, and efficient point cloud alignment solver for accurate global shape learning, and a multi-scale local geometry loss promoting precise local geometry supervision. We train our model on a large, mixed dataset and demonstrate its strong generalizability and high accuracy. In our comprehensive evaluation on diverse unseen datasets, our model significantly outperforms state-of-the-art methods across all tasks, including monocular estimation of 3D point map, depth map, and camera field of view. Code and models will be released on our project page.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.19115) | [🌐 Project Page](https://wangrc.site/MoGePage/) | [💻 Code](https://github.com/microsoft/moge) | [🎮 Demo](https://huggingface.co/spaces/Ruicheng/MoGe)


<br>


## Dynamic Scene Reconstruction:
## 2024:
### 1. MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, Ming-Hsuan Yang
<details span>
<summary><b>Abstract</b></summary>
Estimating geometry from dynamic scenes, where objects move and deform over time, remains a core challenge in computer vision. Current approaches often rely on multi-stage pipelines or global optimizations that decompose the problem into subtasks, like depth and flow, leading to complex systems prone to errors. In this paper, we present Motion DUSt3R (MonST3R), a novel geometry-first approach that directly estimates per-timestep geometry from dynamic scenes. Our key insight is that by simply estimating a pointmap for each timestep, we can effectively adapt DUST3R's representation, previously only used for static scenes, to dynamic scenes. However, this approach presents a significant challenge: the scarcity of suitable training data, namely dynamic, posed videos with depth labels. Despite this, we show that by posing the problem as a fine-tuning task, identifying several suitable datasets, and strategically training the model on this limited data, we can surprisingly enable the model to handle dynamics, even without an explicit motion representation. Based on this, we introduce new optimizations for several downstream video-specific tasks and demonstrate strong performance on video depth and camera pose estimation, outperforming prior work in terms of robustness and efficiency. Moreover, MonST3R shows promising results for primarily feed-forward 4D reconstruction.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.03825) | [🌐 Project Page](https://monst3r-project.github.io/) | [💻 Code](https://github.com/Junyi42/monst3r)

<br>

### 2. Align3R: Aligned Monocular Depth Estimation for Dynamic Videos ![](https://img.shields.io/badge/2024-arXiv-red)
**Authors**: Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, Yuan Liu
<details span>
<summary><b>Abstract</b></summary>
Recent developments in monocular depth estimation methods enable high-quality depth estimation of single-view images but fail to estimate consistent video depth across different frames. Recent works address this problem by applying a video diffusion model to generate video depth conditioned on the input video, which is training-expensive and can only produce scale-invariant depth values without camera poses. In this paper, we propose a novel video-depth estimation method called Align3R to estimate temporal consistent depth maps for a dynamic video. Our key idea is to utilize the recent DUSt3R model to align estimated monocular depth maps of different timesteps. First, we fine-tune the DUSt3R model with additional estimated monocular depth as inputs for the dynamic scenes. Then, we apply optimization to reconstruct both depth maps and camera poses. Extensive experiments demonstrate that Align3R estimates consistent video depth and camera poses for a monocular video with superior performance than baseline methods.
</details>

  [📄 Paper](https://arxiv.org/pdf/2412.03079) | [🌐 Project Page](https://igl-hkust.github.io/Align3R.github.io/) | [💻 Code](https://github.com/jiah-cloud/Align3R)

<br>





## Scene Understanding:
## 2024:
### 1. LargeSpatialModel: End-to-end Unposed Images to Semantic 3D ![](https://img.shields.io/badge/2024-Neurips-blue)
**Authors**: Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, Boris Ivanovic, Marco Pavone, Yue Wang
<details span>
<summary><b>Abstract</b></summary>
Reconstructing and understanding 3D structures from a limited number of images is a well-established problem in computer vision. Traditional methods usually break this task into multiple subtasks, each requiring complex transformations between different data representations. For instance, dense reconstruction through Structure-from-Motion (SfM) involves converting images into key points, optimizing camera parameters, and estimating structures. Afterward, accurate sparse reconstructions are required for further dense modeling, which is subsequently fed into task-specific neural networks. This multi-step process results in considerable processing time and increased engineering complexity.
In this work, we present the Large Spatial Model (LSM), which processes unposed RGB images directly into semantic radiance fields. LSM simultaneously estimates geometry, appearance, and semantics in a single feed-forward operation, and it can generate versatile label maps by interacting with language at novel viewpoints. Leveraging a Transformer-based architecture, LSM integrates global geometry through pixel-aligned point maps. To enhance spatial attribute regression, we incorporate local context aggregation with multi-scale fusion, improving the accuracy of fine local details. To tackle the scarcity of labeled 3D semantic data and enable natural language-driven scene manipulation, we incorporate a pre-trained 2D language-based segmentation model into a 3D-consistent semantic feature field. An efficient decoder then parameterizes a set of semantic anisotropic Gaussians, facilitating supervised end-to-end learning. Extensive experiments across various tasks show that LSM unifies multiple 3D vision tasks directly from unposed images, achieving real-time semantic 3D reconstruction for the first time.
</details>

  [📄 Paper](https://arxiv.org/pdf/2410.18956) | [💻 Code (to be released)]() | [🌐 Project Page](https://largespatialmodel.github.io/) | [🎮 Demo](https://huggingface.co/spaces/kairunwen/LSM)


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
