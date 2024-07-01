# DELE_CA2

The repository for Deep Learning Continous Assement 2. This is a pair work.

# CA2 Instruction

## Guidelines

1. You are to work in groups of two for Parts A and B. For Part C, this is an individual task.
2. In this assignment, you will:
   1. Create GAN for image generation and evaluate the performance of the network.
   2. Create RL to solve the task at hand.
   3. Carry out some literature research and prepare a technical paper on CNN, RNN, GAN/VAE or RL. (Just ONE topic.)
3. For Parts A and B, you should prepare the following:

   1. Jupyter notebook including your code, comments and visualisations (.ipynb).
   2. In addition, please save a copy of the Jupyter notebook as a .html file.
   3. Include your best neural network weights (.h5 file).
   4. A deck of presentation slides (.pptx file) for your project.
   5. A statement indicating the contributions by each member of the group, including percentage of workload for specific contributions.

   NOTE: Each student must be well-versed with both Parts A and B. Whilst you can consider splitting the workload, you will still be evaluated on everything in this CA. You cannot expect to only do one part but know little or nothing about the other part.

4. For Part C, you should submit your file as a Word (.docx) or .pdf document. If you write your own codes, you must include them.
5. Submit all materials in a zipped file. Each student must submit his/her own complete set of files, even if you are sharing for Parts A and B. Any missing files (like missing .h5 file for best weights, missing .html file for your code, etc.) would incur marks deduction.
6. The normal SP’s academic policies on Copyright and Plagiarism applies. Please note that you are to cite all sources. You may refer to the citation guide available at: https://sp-sg.libguides.com/citation.
7. You need to submit your declaration of academic integrity. You may access
   this document on Brightspace. Without this, your submission is deemed
   incomplete.

## Submission Details

**Deadline**: _5 Aug 2024, 08:00 AM_

# PART A: GAN

Apply some suitable GAN architectures to the problem of image generation.
Use the given dataset to create 260 small black-and-white images.
There should be 26 classes of images for you to generate.
You must submit your generated images.

You should implement some ways to evaluate the quality of your images:
_A simple “by-eye” or “eye-power” method is to just generate say 50 images (or 49 = 7 by 7 grid), and count manually how many are acceptable images (not perfect, but display some plausible features of the images --- perhaps as "clear", "marginal", "nonsense"). Alternatively, you can consider other metrics/indicators._

If you are asked to generate images of a specific class, propose a way of doing it.
If you are asked to generate coloured images instead of black-and-white ones, do you think it would be easier or harder to produce better quality results?

What class(es) is/are relatively easier/harder to generate? Why?

Note: Do not worry too much about generating “perfect-looking images”.
Whilst quality of images would represent the quality of your model, the more important aspects are the process, your workflow and planning, your EDA, evaluation analysis, etc., as a whole.

As this is a pair-work, you should discuss and optimise the GAN training between yourselves, making use of available computing resources for both of you.

# PART B: RL

Apply a suitable modification of deep Q-network (DQN) architecture to the problem.
Your model should exert some appropriate torque on the pendulum to balance it.

**<ins>You must use DQN and satisfactorily demonstrate its viability.</ins>**

You may consider other reinforcement learning architectures, if you wish, but only after successfully implementing DQN.
Otherwise, any other non-DQN architecture will be rejected.

In your work, you should plan clearly what you are doing, your approaches, and how you systematically optimise your solutions.

For example, what hyperparameters can you tune? Is one trial enough or should you repeat the trials? Why?

How do you conclusively demonstrate your so-called “best setup” to be the best?
Are you considering fastest learning, most stable learning, or some other criteria that you choose to define?

# PART C: Technical Paper

This part of the assignment is to be completed individually.
This is a challenge task for students who wish to attempt it for higher marks.

Write a technical paper in single column format on any ONE of the following topics:

1. CNN (computer vision, in general)
2. RNN (natural language processing, in general)
3. GAN, VAE (generative AI, in general)
4. RL (reinforcement learning, in general)

In our lessons, we covered some fundamental and basic architectures for these topics.
However, there are certainly more advanced models available beyond what we covered.
Take this opportunity to dive deeper into such extensions.

Here are some examples (but you are free to do other topics not listed here):

_There is a raft of advanced CNN models: ResNet, Inception, VGG. Consider focusing on one such interesting models to you, and explore deeper._

_For RNN, you may delve into how ChatGPT works, the underlying architecture, large language models, for example._

_For GAN, there are various other improvements apart from the basic DCGAN. Even VAE has extensions to areas like physics._

_For RL, we only covered DQN. How can we improve DQN? Double DQN, duelling, priority sampling, etc. How about other RL architectures beyond DQN?_

The paper should have the following components:

1. Abstract
2. Introduction
3. Related Works
4. Dataset/Methodology/Experiment
5. Discussion
6. Conclusions
7. References

Submit the paper in Word or PDF format (page limit of 10 pages).
If you write your own codes, you must submit them to support your work.

# Work Distribution

## Part A

|        TASK         | Irman | Adam | Joachim |
| :-----------------: | :---: | :--: | :-----: |
| Background Research |  0%   | 50%  |   50%   |
|         EDA         |  33%  | 33%  |   33%   |
|   Model Research    |  50%  | 50%  |   0%    |
|  Initial Modelling  |  70%  |  0%  |   30%   |
|  Model Improvement  |  33%  | 33%  |   33%   |
|  Model Evaluation   |  40%  | 20%  |   40%   |
|   Report/Markdown   |  33%  | 33%  |   33%   |

## Part B

|        TASK         | Irman | Adam | Joachim |
| :-----------------: | :---: | :--: | :-----: |
| Background Research |  0%   | 50%  |   50%   |
|         EDA         |  33%  | 33%  |   33%   |
|   Model Research    |  50%  | 50%  |   0%    |
|  Initial Modelling  |  70%  |  0%  |   30%   |
|  Model Improvement  |  33%  | 33%  |   33%   |
|  Model Evaluation   |  40%  | 20%  |   40%   |
|   Report/Markdown   |  33%  | 33%  |   33%   |
