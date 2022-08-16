# READING LIST

:fire: :collision: :anger: :exclamation: :cyclone: :ocean: :sweat_drops: :droplet: :turtle: :frog: :snake: :bug: :ram: :sheep: :rabbit2: :goat: :whale: :whale2: :fish: :dolphin: :rose: :maple_leaf: :fallen_leaf: :mushroom: :beer: :beers: :airplane: :fountain: :bullettrain_front: :rocket: :rowboat: :speedboat:

:ear_of_rice: :pager: :page_facing_up: :page_with_curl: :book: :blue_book: :bookmark: :bookmark_tabs: :books: :clipboard: :bar_chart: :chart_with_downwards_trend: :chart_with_upwards_trend: :card_file_box: :card_index: :card_index_dividers: :flower_playing_cards: :newspaper: :newspaper_roll: :closed_book: :orange_book: :notebook: :notebook_with_decorative_cover: :ledger: :spiral_notepad: :green_book: :scroll: :pushpin: :fountain_pen: :pen: :pencil2: :santa:

Recommendation: :+1: :fire: :volcano: :boom:

To-Do (Reading) List: :droplet: :sweat_drops: [并没有说水的意思😐🙏] 

TOC  

- [READING LIST](#reading-list)
  - [Emp. & ASP](#emp--asp)
  - [Meta-RL](#meta-rl)
  - [HRL](#hrl)
  - [SKILLS](#skills)
  - [Control as Inference](#control-as-inference)
  - [State Abstraction, Representation Learning](#state-abstraction-representation-learning)
  - [Mutual Information](#mutual-information)
  - [DR (Domain Randomization) & sim2real](#dr-domain-randomization--sim2real)
  - [Transfer: Generalization & Adaption (Dynamics)](#transfer-generalization--adaption-dynamics)
  - [IL (IRL)](#il-irl)
  - [Offline RL](#offline-rl)
  - [Exploration](#exploration)
  - [Causal Inference](#causal-inference)
  - [Supervised RL & Goal-conditioned Policy](#supervised-rl--goal-conditioned-policy)
  - [Goal-relabeling & Self-imitation](#goal-relabeling--self-imitation)
  - [Model-based RL & world models](#model-based-rl--world-models)
  - [Training RL & Just Fast & Embedding? & OPE(DICE)](#training-rl--just-fast--embedding--opedice)
  - [MARL](#marl)
  - [Constrained RL](#constrained-rl)
  - [Distributional RL](#distributional-rl)
  - [Continual Learning](#continual-learning)
  - [Self-paced & Curriculum RL](#self-paced--curriculum-rl)
  - [Quadruped](#quadruped)
  - [Optimization](#optimization)
  - [Galaxy  Forest](#galaxy--forest)
  - [Aha](#aha)
    - [Alpha](#alpha)
    - [Blog & Corp. & Legend](#blog--corp--legend)

<a name="anchor-emp"></a>  

## Emp. & ASP

:partly_sunny: :full_moon: :shell: :seedling: :evergreen_tree: :mushroom: :leaves: :sunflower: :cactus: :bouquet: :herb: :palm_tree: :four_leaf_clover: :deciduous_tree: :rose: :chestnut: :bamboo: :ear_of_rice: :palm_tree: :maple_leaf: :paw_prints: :new_moon: :first_quarter_moon: :waning_gibbous_moon: :waning_crescent_moon: :full_moon: :milky_way: :globe_with_meridians: :earth_americas: :volcano: :jack_o_lantern:

- Empowerment — An Introduction <https://arxiv.org/pdf/1310.1863.pdf> :+1:

- Keep your options open: An information-based driving principle for sensorimotor systems  

  - It measures the capacity of the agent to influence the world in a way that this influence is perceivable via the agent’s sensors.
  - Concretely, we define empowerment as the maximum amount of information that an agent could send from its actuators to its sensors via the environment, reducing in the simplest case to the external information channel capacity of the channel from the actuators to the sensors of the agent.

  - An individual agent or an agent population can attempt and explore only a small fraction of possible behaviors during its lifetime.

  - universal & local

- What is intrinsic motivation? A typology of computational approaches

- [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning[2015]](https://arxiv.org/abs/1509.08731)  :+1:​

  We focussed specifically on intrinsic motivation with a reward measure known as empowerment, which requires at its core the efficient computation of the mutual information.

- Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning

- A survey on intrinsic motivation in reinforcement learning <https://arxiv.org/abs/1908.06976> :milky_way: :fist_oncoming: :fist_oncoming: :fire: :fire: :sweat_drops: :droplet: :droplet:

- Efficient Exploration via State Marginal Matching <https://arxiv.org/pdf/1906.05274.pdf> :volcano:

- Empowerment: A Universal Agent-Centric Measure of Control <https://uhra.herts.ac.uk/bitstream/handle/2299/1114/901241.pdf?sequence=1&isAllowed=y>

  🔹 [On Learning Intrinsic Rewards for Policy Gradient Methods](https://arxiv.org/pdf/1804.06459.pdf) :fire:

  The policy-gradient updates the policy parameters to optimize the sum of the extrinsic and intrinsic rewards, while simultaneously our method updates the intrinsic reward parameters to optimize the extrinsic rewards achieved by the policy.

  🔹 [Adversarial Intrinsic Motivation for Reinforcement Learning](https://arxiv.org/pdf/2105.13345.pdf) :droplet:

  🔹 [Evaluating Agents without Rewards](https://arxiv.org/pdf/2012.11538.pdf) :no_mouth:

  We retrospectively compute potential objectives on pre-collected datasets of agent behavior, rather than optimizing them online, and compare them by analyzing their correlations.

  🔹 [LEARNING ALTRUISTIC BEHAVIOURS IN REINFORCEMENT LEARNING WITHOUT EXTERNAL REWARDS](https://openreview.net/pdf?id=KxbhdyiPHE) :fire:

  We propose an altruistic agent that learns to increase the choices another agent has by preferring to maximize the number of states that the other agent can reach in its future.  

  🔹 [Entropic Desired Dynamics for Intrinsic Control](https://proceedings.neurips.cc/paper/2021/file/5f7f02b7e4ade23430f345f954c938c1-Paper.pdf) :fire:  

  EDDICT:  By situating these latent codes in a globally consistent coordinate system, we show that agents can reliably reach more states in the long term while still optimizing a local objective.  

- SMiRL: Surprise Minimizing Reinforcement Learning in Dynamic Environments <https://openreview.net/pdf?id=cPZOyoDloxl>  :fire: :boom: :volcano: :boom: :fire: :droplet:  

  In the real world, natural forces and other agents already **offer unending novelty**. The second law of thermodynamics stipulates **ever-increasing entropy**, and therefore perpetual novelty, without even requiring any active intervention.  

  🔹 [Unsupervised Skill Discovery with Bottleneck Option Learning](https://arxiv.org/pdf/2106.14305.pdf) :+1: :fire:  

  On top of the linearization of environments that promotes more various and distant state transitions, IBOL enables the discovery of diverse skills.

  🔹 [Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions](https://arxiv.org/pdf/1901.01753.pdf)  ​​ [Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions](https://arxiv.org/pdf/2003.08536.pdf) :no_mouth:  ​
  
  🔹 [The Viable System Model - Stafford Beer](https://www.businessballs.com/strategy-innovation/viable-system-model-stafford-beer/)  
  
  🔹 [Reinforcement Learning Generalization with Surprise Minimization](https://arxiv.org/pdf/2004.12399.pdf) :no_mouth:  ​
  
  🔹 [TERRAIN RL SIMULATOR](https://arxiv.org/pdf/1804.06424.pdf)   [Github](https://github.com/UBCMOCCA/TerrainRLSim)  

  🔹 [POLTER: Policy Trajectory Ensemble Regularization for Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2205.11357.pdf) :no_mouth:  

  It utilizes an ensemble of policies that are discovered during pretraining and moves the policy of the URL algorithm closer to its optimal prior.  

  🔹 [WALK THE RANDOM WALK: LEARNING TO DISCOVER AND REACH GOALS WITHOUT SUPERVISION](https://arxiv.org/pdf/2206.11733.pdf) 😶 

  We use random walk to train a reachability network that predicts the similarity between two states. This reachability network is then used in building goal memory containing past observations that are diverse and well-balanced. Finally, we train a goal-conditioned policy network with goals sampled from the goal memory and reward it by the reachability network and the goal memory.


<a name="anchor-asp"></a>  

- ASP: ASYMMETRIC SELF-PLAY

  🔹 INTRINSIC MOTIVATION AND AUTOMATIC CURRICULA VIA ASYMMETRIC SELF-PLAY <https://arxiv.org/pdf/1703.05407.pdf> [起飞 ASP] :fire: :fire: :+1:

  🔹 [Keeping Your Distance: Solving Sparse Reward Tasks Using Self-Balancing Shaped Rewards](https://papers.nips.cc/paper/9225-keeping-your-distance-solving-sparse-reward-tasks-using-self-balancing-shaped-rewards.pdf) [ASP] :fire: :volcano: :+1:

  Our method introduces an auxiliary distance-based reward based on pairs of rollouts to encourage diverse exploration. This approach effectively prevents learning dynamics from stabilizing around local optima induced by the naive distance-to-goal reward shaping and enables policies to efficiently solve sparse reward tasks.

  🔹 [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)  :droplet:

  🔹 [Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1811.09083.pdf) [ASP] :fire: :+1:

  🔹 [Generating Automatic Curricula via Self-Supervised Active Domain Randomization](https://arxiv.org/pdf/2002.07911.pdf) [ASP]

  🔹[ASYMMETRIC SELF-PLAY FOR AUTOMATIC GOAL DISCOVERY IN ROBOTIC MANIPULATION](https://openreview.net/pdf?id=hu2aMLzOxC) :no_mouth:  [ASP]

  🔹 [Language as a Cognitive Tool to Imagine Goals in Curiosity-Driven Exploration](https://arxiv.org/pdf/2002.09253.pdf) :fire: :boom:  ​ ​
  
  We introduce IMAGINE, an intrinsically motivated deep reinforcement learning architecture that models this ability. Such imaginative agents, like children, benefit from the guidance of a social peer who provides language descriptions. To take advantage of goal imagination, agents must be able to leverage these descriptions to interpret their imagined out-of-distribution goals.

+ PBRL (Population Based); Quality-Diversity (QD); 
  
  🔹 [Effective Diversity in Population Based Reinforcement Learning](https://arxiv.org/pdf/2002.00632.pdf) 😶 

  Diversity via Determinants (DvD)

  🔹 [“Other-Play” for Zero-Shot Coordination](https://arxiv.org/pdf/2003.02979.pdf) 

  zero-shot coordination: constructing AI agents that can coordinate with novel partners they have not seen before. Other-Play (OP) enhances self-play by looking for more robust strategies, exploiting the presence of known symmetries in the underlying problem.

  🔹 [Trajectory Diversity for Zero-Shot Coordination](http://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf) 

  TrajeDi: a differentiable objective for generating diverse reinforcement learning policies.

  🔹 [Illuminating search spaces by mapping elites](https://arxiv.org/pdf/1504.04909.pdf) 🌋 :fire: 

  Multi-dimensional Archive of Phenotypic Elites (MAP-Elites) algorithm illuminates search spaces, allowing researchers to understand how interesting attributes of solutions combine to affect performance, either positively or, equally of interest, negatively.

  🔹 [Differentiable Quality Diversity](https://proceedings.neurips.cc/paper/2021/file/532923f11ac97d3e7cb0130315b067dc-Paper.pdf) :+1: 

  We present the differentiable quality diversity (DQD) problem, a special case of QD, where both the objective and measure functions are first order differentiable. 

  🔹 [Diversity Policy Gradient for Sample Efficient Quality-Diversity Optimization](https://arxiv.org/pdf/2006.08505.pdf) :fire: 

  qd-pg: The main contribution of this work is the introduction of a Diversity Policy Gradient (DPG) that exploits information at the time-step level to drive policies towards more diversity in a sample efficient manner.

  🔹 [Discovering Diverse Nearly Optimal Policies with Successor Features](https://arxiv.org/pdf/2106.00669.pdf) :fire: :fire: 

  we propose new explicit diversity rewards that aim to minimize the correlation between the Successor Features of the policies in the set.

  🔹 [Continual Auxiliary Task Learning](https://proceedings.neurips.cc/paper/2021/file/68331ff0427b551b68e911eebe35233b-Paper.pdf) 🌋 

  we investigate a reinforcement learning system designed to learn a collection of auxiliary tasks, with a behavior policy learning to take actions to improve those auxiliary predictions.

  🔹 [Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games](https://proceedings.neurips.cc/paper/2021/file/07bba581a2dd8d098a3be0f683560643-Paper.pdf) 🌋 

  we summarize previous concepts of diversity and work towards offering a unified measure of diversity in multi-agent open-ended learning to include all elements in Markov games, based on both Behavioral Diversity (BD) and Response Diversity (RD).

  🔹 [CONTINUOUSLY DISCOVERING NOVEL STRATEGIES VIA REWARD-SWITCHING POLICY OPTIMIZATION](https://arxiv.org/pdf/2204.02246.pdf) :fire: 🌋 

  RSPO: When a sampled trajectory is sufficiently distinct, RSPO performs standard policy optimization with extrinsic rewards. For trajectories with high likelihood under existing policies, RSPO utilizes an intrinsic diversity reward to promote exploration.

  🔹 [DGPO: Discovering Multiple Strategies with Diversity-Guided Policy Optimization](https://arxiv.org/pdf/2207.05631.pdf) :fire: 

  we formalize our algorithm as the combination of a diversity-constrained optimization problem and an extrinsic-reward constrained optimization problem. 

  🔹 [POPULATION-GUIDED PARALLEL POLICY SEARCH FOR REINFORCEMENT LEARNING](https://arxiv.org/pdf/2001.02907.pdf) :fire: 🌋 

  P3S: The key point is that the information of the best policy is fused in a soft manner by constructing an augmented loss function for policy update to enlarge the overall search region by the multiple learners. 

  🔹 [Periodic Intra-Ensemble Knowledge Distillation for Reinforcement Learning](https://arxiv.org/pdf/2002.00149.pdf) 😶 

   PIEKD is a learning framework that uses an ensemble of policies to act in the environment while periodically sharing knowledge amongst policies in the ensemble through knowledge distillation.

   🔹 [Cooperative Heterogeneous Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/ca3a9be77f7e88708afb20c8cdf44b60-Paper.pdf) :fire:

  CHDRL: Global agents are off-policy agents that can utilize experiences from the other agents. Local agents are either on-policy agents or population-based evolutionary algorithms (EAs) agents that can explore the local area effectively.

  🔹 [Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks](https://arxiv.org/pdf/2004.05937.pdf) 



  🔹 [General Characterization of Agents by States they Visit](https://arxiv.org/pdf/2012.01244.pdf) :+1: 

  Behavioural characterizations: adopt Gaussian mixture models (GMMs). 

  🔹 [Improving Policy Optimization with Generalist-Specialist Learning](http://ai.ucsd.edu/~haosu/Other_Doc/GSL.pdf) 😶 

  GSL: we first train a generalist on all environment variations; when it fails to improve, we launch a large population of specialists with weights cloned from the generalist, each trained to master a selected small subset of variations. We finally resume the training of the generalist with auxiliary rewards induced by demonstrations of all specialists.

  🔹 [Diversity Can Be Transferred: Output Diversification for White- and Black-box Attacks](https://proceedings.neurips.cc/paper/2020/file/30da227c6b5b9e2482b6b221c711edfd-Paper.pdf) :fire: 🌋 🌋 

  Output Diversified Sampling (ODS): a novel sampling strategy that attempts to maximize diversity in the target model’s outputs among the generated samples.

  🔹 [Diversity Matters When Learning From Ensembles](https://proceedings.neurips.cc/paper/2021/file/466473650870501e3600d9a1b4ee5d44-Paper.pdf) :fire: 

  Our key assumption is that a distilled model should absorb as much function diversity inside the ensemble as possible.

  🔹 [Improving Ensemble Distillation With Weight Averaging and Diversifying Perturbation](https://arxiv.org/pdf/2206.15047.pdf) :fire: 

  we propose a weight averaging technique where a student with multiple subnetworks is trained to absorb the functional diversity of ensemble teachers, but then those subnetworks are properly averaged for inference, giving a single student network with no additional inference cost. We also propose a perturbation strategy that seeks inputs from which the diversities of teachers can be better transferred to the student.



  
<a name="anchor-metarl"></a>  

## Meta-RL

 :frog: :tiger: :snail: :snake: :camel: :tiger: :turtle: :bird: :ant: :koala: :dog: :beetle: :chicken: :rat: :ox: :cow2: :whale: :fish: :dolphin: :whale2: :cat2: :blowfish: :dragon: :dragon_face: :goat: :octopus: :ant: :turtle: :crocodile: :baby_chick:

- A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms 2020  <https://arxiv.org/pdf/1901.10912.pdf> Yoshua Bengio :milky_way: :fire: :fire: :+1: :maple_leaf: :sweat_drops:  [contrative loss on causal mechanisms?]

  We show that under this assumption, the correct causal structural choices lead to faster adaptation to modified distributions because the changes are concentrated in one or just a few mechanisms when the learned knowledge is modularized appropriately.

- Causal Reasoning from Meta-reinforcement Learning 2019 :wink: :no_mouth:

- Discovering Reinforcement Learning Algorithms <https://arxiv.org/pdf/2007.08794.pdf> :+1:

  This paper introduces a new meta-learning approach that discovers an entire update rule which includes both ‘**what to predict**’ (e.g. value functions) and ‘**how to learn from it**’ (e.g. bootstrapping) by interacting with a set of environments.

- Meta

  🔹 [Discovering Reinforcement Learning Algorithms](https://arxiv.org/pdf/2007.08794.pdf)   Attempte to discover the full update rule :+1: ​

  🔹 [What Can Learned Intrinsic Rewards Capture?](https://arxiv.org/pdf/1912.05500.pdf)   How/What value function/policy network :+1:  

  ​ lifetime return:A finite sequence of agent-environment interactions until the end of training defined by an agentdesigner, which can consist of multiple episodes.

  🔹 [Discovery of Useful Questions as Auxiliary Tasks](http://papers.nips.cc/paper/9129-discovery-of-useful-questions-as-auxiliary-tasks.pdf) :confused:

  ​ Related work is good! (Prior work on auxiliary tasks in RL + GVF) :fire:  :+1:

  🔹 [Meta-Gradient Reinforcement Learning](http://papers.nips.cc/paper/7507-meta-gradient-reinforcement-learning.pdf)  discount factor + bootstrapped factor :sweat_drops:  ​

  🔹 [BEYOND EXPONENTIALLY DISCOUNTED SUM: AUTOMATIC LEARNING OF RETURN FUNCTION](https://arxiv.org/pdf/1905.11591.pdf) :no_mouth:

  We research how to modify the form of the return function to enhance the learning towards the optimal policy. We propose to use a general mathematical form for return function, and employ meta-learning to learn the optimal return function in an end-to-end manner.

  🔹 [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf) :fire: :volcano: :boom:

  MAML:  In our approach, the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task.

  🔹 [BERT Learns to Teach: Knowledge Distillation with Meta Learning](https://arxiv.org/pdf/2106.04570.pdf) :volcano:

  MetaDistill: We show the teacher network can learn to better transfer knowledge to the student network (i.e., learning to teach) with the feedback from the performance of the distilled student network in a meta learning framework.  

  🔹 [Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](http://proceedings.mlr.press/v97/rakelly19a/rakelly19a.pdf) :fire: :fire:

  PEARL: Current methods rely heavily on on-policy experience, limiting their sample efficiency. They also lack mechanisms to reason about task uncertainty when adapting to new tasks, limiting their effectiveness on sparse reward problems. We address these challenges by developing an offpolicy meta-RL algorithm that disentangles task inference and control.

  🔹 [Guided Meta-Policy Search](https://arxiv.org/pdf/1904.00956.pdf) :+1: :fire: :volcano:  

  GMPS: We propose to learn a RL procedure in a federated way, where individual off-policy learners can solve the individual meta-training tasks, and then consolidate these solutions into a single meta-learner. Since the central meta-learner learns by imitating the solutions to the individual tasks, it can accommodate either the standard meta-RL problem setting, or a hybrid setting where some or all tasks are provided with example demonstrations.

  🔹 [CoMPS: Continual Meta Policy Search](https://arxiv.org/pdf/2112.04467.pdf) :fire: 

  CoMPS continuously repeats two subroutines: learning a new task using RL and using the experience from RL to perform completely offline meta-learning to prepare for subsequent task learning.

  🔹 [Bootstrapped Meta-Learning](https://arxiv.org/pdf/2109.04504.pdf) :fire: :volcano:

  We propose an algorithm that tackles these issues by letting the metalearner teach itself. The algorithm first bootstraps a target from the meta-learner, then optimises the meta-learner by minimising the distance to that target under a chosen (pseudo-)metric.

  🔹 [Taming MAML: Efficient Unbiased Meta-Reinforcement Learning](http://proceedings.mlr.press/v97/liu19g/liu19g.pdf) :+1: :fire:

  TMAML: that adds control variates into gradient estimation via automatic differentiation. TMAML improves the quality of gradient estimation by reducing variance without introducing bias.

  🔹 [NoRML: No-Reward Meta Learning](https://arxiv.org/pdf/1903.01063.pdf) :+1:

  NoRML: The key insight underlying NoRML is that we can simultaneously learn the meta-policy and the advantage function used for adapting the meta-policy, optimizing for the ability to effectively adapt to varying dynamics.

  🔹 [SKILL-BASED META-REINFORCEMENT LEARNING](https://openreview.net/pdf?id=jeLW-Fh9bV) :+1: :fire:

  we propose to (1) extract reusable skills and a skill prior from offline datasets, (2) meta-train a high-level policy that learns to efficiently compose learned skills into long-horizon behaviors, and (3) rapidly adapt the meta-trained policy to solve an unseen target task.

  🔹 [Offline Meta Learning of Exploration](https://arxiv.org/pdf/2008.02598.pdf) :droplet:

  We take a Bayesian RL (BRL) view, and seek to learn a Bayes-optimal policy from the offline data. Building on the recent VariBAD BRL approach, we develop an off-policy BRL method that learns to plan an exploration strategy based on an adaptive neural belief estimate.

- Unsupervised Meta-Learning for Reinforcement Learning <https://arxiv.org/pdf/1806.04640.pdf> [Abhishek Gupta, Benjamin Eysenbach, Chelsea Finn, Sergey Levine] :confused: :wink:

  Meta-RL shifts the human burden from algorithm to task design. In contrast, our work deals with the RL setting, where the environment dynamics provides a rich inductive bias that our meta-learner can exploit.

  🔹 [UNSUPERVISED LEARNING VIA META-LEARNING](https://arxiv.org/pdf/1810.02334.pdf)  :wink:  ​We construct tasks from unlabeled data in an automatic way and run meta-learning over the constructed tasks.

  🔹 [Unsupervised Curricula for Visual Meta-Reinforcement Learning](https://arxiv.org/pdf/1912.04226.pdf)  [Allan Jabriα; Kyle Hsu] :+1: :droplet: :volcano: :fire:

  Yet, the aforementioned relation between skill acquisition and meta-learning suggests that they should not be treated separately.

  However, relying solely on discriminability becomes problematic in environments with high-dimensional (image-based) observation spaces as it **results in an issue akin to mode-collapse in the task space**. This problem is further complicated in the setting we propose to study, wherein the policy data distribution is that of a meta-learner rather than a contextual policy. We will see that this can be ameliorated by specifying **a hybrid discriminative-generative model** for parameterizing the task distribution.

  We, rather, will **tolerate lossy representations** as long as they capture discriminative features useful for stimulus-reward association.  

  🔹 [On the Effectiveness of Fine-tuning Versus Meta-reinforcement Learning](https://arxiv.org/pdf/2206.03271.pdf) 😶

  Conclusion: multi-task pretraining with fine-tuning on new tasks performs equally as well, or better, than meta-RL.

- Asymmetric Distribution Measure for Few-shot Learning <https://arxiv.org/pdf/2002.00153.pdf> :+1:

  feature representations and relation measure.

- latent models

  🔹 [MELD: Meta-Reinforcement Learning from Images via Latent State Models](https://arxiv.org/pdf/2010.13957.pdf) :+1: :+1:  ​ ​

  we leverage the perspective of meta-learning as task inference to show that latent state models can also perform meta-learning given an appropriately defined observation space.

  🔹 [Explore then Execute: Adapting without Rewards via Factorized Meta-Reinforcement Learning](https://arxiv.org/pdf/2008.02790.pdf) :+1: :+1:

  based on identifying key information in the environment, independent of how this information will exactly be used solve the task. By decoupling exploration from task execution, DREAM explores and consequently adapts to new environments, requiring no reward signal when the task is specified via an instruction.  

- model identification and experience relabeling (MIER)

  🔹 [Meta-Reinforcement Learning Robust to Distributional Shift via Model Identification and Experience Relabeling](https://arxiv.org/pdf/2006.07178.pdf) :+1: :fire:  ​ ​

  Our method is based on a simple insight: we recognize that dynamics models can be adapted efficiently and consistently with off-policy data, more easily than policies and value functions. These dynamics models can then be used to continue training policies and value functions for out-of-distribution tasks without using meta-reinforcement learning at all, by generating synthetic experience for the new task.

<a name="anchor-HRL"></a>  

## HRL

- SUB-POLICY ADAPTATION FOR HIERARCHICAL REINFORCEMENT LEARNING <https://arxiv.org/pdf/1906.05862.pdf> :-1:  

  🔹 [STOCHASTIC NEURAL NETWORKS FOR HIERARCHICAL REINFORCEMENT LEARNING](https://arxiv.org/pdf/1704.03012.pdf)  

- HIERARCHICAL RL USING AN ENSEMBLE OF PROPRIOCEPTIVE PERIODIC POLICIES <https://openreview.net/pdf?id=SJz1x20cFQ> :-1:

- LEARNING TEMPORAL ABSTRACTION WITH INFORMATION-THEORETIC CONSTRAINTS FOR HIERARCHICAL REINFORCEMENT LEARNING <https://openreview.net/pdf?id=HkeUDCNFPS> :fire: :+1:

  we maximize the mutual information between the latent variables and the state changes.

  🔹 [Hierarchical Reinforcement Learning with Advantage-Based Auxiliary Rewards](https://proceedings.neurips.cc/paper/2019/file/81e74d678581a3bb7a720b019f4f1a93-Paper.pdf) :fire: 

  HAAR: We propose an HRL framework which sets auxiliary rewards for low-level skill training based on the advantage function of the high-level policy.
  
- learning representation

  🔹 [LEARNING SUBGOAL REPRESENTATIONS WITH SLOW DYNAMICS](https://openreview.net/pdf?id=wxRwhSdORKG) :+1: :droplet: :fire: ​​  ​ ​

  Observing that the high-level agent operates at an abstract temporal scale, we propose a slowness objective to effectively learn the subgoal representation (i.e., the high-level action space). We provide a theoretical grounding for the slowness objective.  ​
  
  🔹 [ACTIVE HIERARCHICAL EXPLORATION WITH STABLE SUBGOAL REPRESENTATION LEARNING](https://openreview.net/pdf?id=sNuFKTMktcY) :+1:
  
  HESS: We propose a novel regularization that contributes to both stable and efficient subgoal representation learning.
  
- meta; skills

  🔹 [LEARNING TRANSFERABLE MOTOR SKILLS WITH HIERARCHICAL LATENT MIXTURE POLICIES](https://arxiv.org/pdf/2112.05062.pdf) :fire:

  our method exploits a three-level hierarchy of both discrete and continuous latent variables, to capture a set of high-level behaviours while allowing for variance in how they are executed.  

  🔹 [Hierarchical Planning Through Goal-Conditioned Offline Reinforcement Learning](https://arxiv.org/pdf/2205.11790.pdf) :fire: 

  HiGoC: The low-level policy is trained via offline RL. We improve the offline training to deal with out-of-distribution goals by a perturbed goal sampling process. The high-level planner selects intermediate sub-goals by taking advantages of model-based planning methods.

  🔹 [Example-Driven Model-Based Reinforcement Learning for Solving Long-Horizon Visuomotor Tasks](https://arxiv.org/pdf/2109.10312.pdf) :+1: :fire: 

  EMBR learns and plans using a learned model, critic, and success classifier, where the success classifier serves both as a reward function for RL and as a grounding mechanism to continuously detect if the robot should retry a skill when unsuccessful or under perturbations.

  🔹 [Planning to Practice: Efficient Online Fine-Tuning by Composing Goals in Latent Space](https://arxiv.org/pdf/2205.08129.pdf) :+1: 

  PTP: first, a high-level planner that sets intermediate subgoals using conditional subgoal generators in the latent space for a lowlevel model-free policy. second, a hybrid approach which first pre-trains both the conditional subgoal generator and the policy on previously collected data through offline reinforcement learning, and then fine-tunes the policy via online.

## SKILLS

- [Latent Space Policies for Hierarchical Reinforcement Learning 2018](https://arxiv.org/pdf/1804.02808.pdf )

- EPISODIC CURIOSITY THROUGH REACHABILITY [reward design]

  In particular, inspired by curious behaviour in animals, observing something novel could be rewarded with a bonus. Such bonus is summed up with the real task reward — making it possible for RL algorithms to learn from the combined reward. We propose a new curiosity method which uses episodic memory to form the novelty bonus. :droplet: **To determine the bonus, the current observation is compared with the observations in memory.** Crucially, the comparison is done based on how many environment steps it takes to reach the current observation from those in memory — which incorporates rich information about environment dynamics. This allows us to overcome the known “couch-potato” issues of prior work — when the agent finds a way to instantly gratify itself by exploiting actions which lead to hardly predictable consequences.

<a name="anchor-comskills"></a>   <a name="anchor-klreg"></a>  

- Combing Skills & **KL regularized expected reward objective**

  🔹 [INFOBOT: TRANSFER AND EXPLORATION VIA THE INFORMATION BOTTLENECK](https://arxiv.org/pdf/1901.10902.pdf) :fire: :boom:  ​ ​

  By training a goal-conditioned policy with an information bottleneck, we can identify decision states by examining where the model actually leverages the goal state.

  🔹 [THE VARIATIONAL BANDWIDTH BOTTLENECK: STOCHASTIC EVALUATION ON AN INFORMATION BUDGET](https://arxiv.org/pdf/2004.11935.pdf) :fire: :boom: :volcano:  

  we propose the variational bandwidth bottleneck, which decides for each example on the estimated value of the privileged information before seeing it, i.e., only based on the standard input, and then accordingly chooses stochastically, whether to access the privileged input or not.

  🔹 [the option keyboard Combing Skills in Reinforcement Learning](https://papers.nips.cc/paper/9463-the-option-keyboard-combining-skills-in-reinforcement-learning.pdf)  

  We argue that a more robust way of combining skills is to do so directly in **the goal space**, using pseudo-rewards or cumulants. If we associate each skill with a cumulant, we can combine the former by manipulating the latter. This allows us to go beyond the direct prescription of behaviors, working instead in the space of intentions. :confused:

  Others: 1. in the space of policies -- over actions; 2. manipulating the corresponding parameters.

  🔹 [Scaling simulation-to-real transfer by learning composable robot skills](https://arxiv.org/pdf/1809.10253.pdf) :fire: :+1: :boom:

  we first use simulation to jointly learn a policy for a set of low-level skills, and a **“skill embedding”** parameterization which can be used to compose them.

  🔹 [LEARNING AN EMBEDDING SPACE FOR TRANSFERABLE ROBOT SKILLS](https://openreview.net/pdf?id=rk07ZXZRb) :fire: :volcano:

  our method is able to learn the skill embedding distributions, which enables interpolation between different skills as well as discovering the number of distinct skills necessary to accomplish a set of tasks.

  🔹 [CoMic: Complementary Task Learning & Mimicry for Reusable Skills](https://proceedings.icml.cc/static/paper_files/icml/2020/5013-Paper.pdf) :fire: :boom:  ​ ​

   We study the problem of learning reusable humanoid skills by imitating motion capture data and joint training with complementary tasks. **Related work is good!**

  🔹 [Learning to combine primitive skills: A step towards versatile robotic manipulation](https://arxiv.org/pdf/1908.00722.pdf) :+1:  ​

  RL(high-level) + IM (low-level)

  🔹 [COMPOSABLE SEMI-PARAMETRIC MODELLING FOR LONG-RANGE MOTION GENERATION](https://openreview.net/pdf?id=rkl44TEtwH) :+1:  ​

  Our proposed method learns to model the motion of human by combining the complementary strengths of both non-parametric techniques and parametric ones. Good EXPERIMENTS!

  🔹 [LEARNING TO COORDINATE MANIPULATION SKILLS VIA SKILL BEHAVIOR DIVERSIFICATION](https://openreview.net/pdf?id=ryxB2lBtvH) :fire: :+1:  ​ ​

  Our method consists of two parts: (1) acquiring primitive skills with diverse behaviors by mutual information maximization, and (2) learning a meta policy that selects a skill for each end-effector and coordinates the chosen skills by controlling the behavior of each skill. **Related work is good!**

  🔹 [Information asymmetry in KL-regularized RL](https://arxiv.org/pdf/1905.01240.pdf) :fire: :boom: :+1:  ​ ​ ​

  In this work we study the possibility of leveraging such repeated structure to speed up and regularize learning. We start from the **KL regularized expected reward objective** which introduces an additional component, a default policy. Instead of relying on a fixed default policy, we learn it from data. But crucially, we **restrict the amount of information the default policy receives**, forcing it to learn reusable behaviours that help the policy learn faster.

  🔹 [Exploiting Hierarchy for Learning and Transfer in KL-regularized RL](https://arxiv.org/pdf/1903.07438.pdf) :+1: :boom: :fire: :droplet:

  The KL-regularized expected reward objective constitutes a convenient tool to this end. It introduces an additional component, a default or prior behavior, which can be learned alongside the policy and as such partially transforms the reinforcement learning problem into one of behavior modelling. **In this work we consider the implications of this framework in case where both the policy and default behavior are augmented with latent variables.** We discuss how the resulting hierarchical structures can be exploited to implement different inductive biases and how the resulting modular structures can be exploited for transfer. Good Writing / Related-work!  :+1:  

  🔹 [CompILE: Compositional Imitation Learning and Execution](http://proceedings.mlr.press/v97/kipf19a/kipf19a.pdf) :no_mouth:  ​

  CompILE can successfully discover sub-tasks and their boundaries in an imitation learning setting.

  🔹 [Strategic Attentive Writer for Learning Macro-Actions](http://papers.nips.cc/paper/6414-strategic-attentive-writer-for-learning-macro-actions.pdf) :no_mouth:  ​

  🔹[Synthesizing Programs for Images using Reinforced Adversarial Learning](https://arxiv.org/pdf/1804.01118.pdf) :no_mouth: RL render RENDERS  ​

  🔹 [Neural Task Graphs: Generalizing to Unseen Tasks from a Single Video Demonstration](https://arxiv.org/pdf/1807.03480.pdf) :fire: :+1:  ​ ​

   The NTG networks consist of a generator that produces the conjugate task graph as the intermediate representation, and an execution engine that executes the graph by localizing node and deciding the edge transition in the task graph based on the current visual observation.

  🔹 [Reinforcement Learning with Competitive Ensembles of Information-Constrained Primitives](https://openreview.net/pdf?id=ryxgJTEYDr) :fire: :volcano:  

  each primitive chooses how much information it needs about the current state to make a decision and the primitive that requests the most information about the current state acts in the world.

  🔹 [COMPOSING TASK-AGNOSTIC POLICIES WITH DEEP REINFORCEMENT LEARNING](https://openreview.net/pdf?id=H1ezFREtwH) :+1:  ​

  🔹 [DISCOVERING A SET OF POLICIES FOR THE WORST CASE REWARD](https://openreview.net/pdf?id=PUkhWz65dy5) :+1:

  the problem we are solving can be seen as the definition and discovery of lower-level policies that will lead to a robust hierarchical agent.

  🔹 [CONSTRUCTING A GOOD BEHAVIOR BASIS FOR TRANSFER USING GENERALIZED POLICY UPDATES](https://openreview.net/pdf?id=7IWGzQ6gZ1D) :fire:

  We show theoretically that, under certain assumptions, having access to a specific set of diverse policies, which we call a set of independent policies, can allow for instantaneously achieving high-level performance on all possible downstream tasks which are typically more complex than the ones on which the agent was trained.

   ​

- Acquiring Diverse Robot Skills via Maximum Entropy Deep Reinforcement Learning [Tuomas Haarnoja, UCB]  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2018/EECS-2018-176.pdf> :fire: :boom: :sweat_drops: :sweat_drops:

- [One Solution is Not All You Need: Few-Shot Extrapolation via Structured MaxEnt RL](https://biases-invariances-generalization.github.io/pdf/big_35.pdf) :+1:  ​  

- [Evaluating Agents without Rewards](https://openreview.net/pdf?id=FoM-RnF6SNe) :+1: ​ :sweat_drops:  ​

- [Explore, Discover and Learn: Unsupervised Discovery of State-Covering Skills](https://arxiv.org/pdf/2002.03647.pdf) :+1: :fire: :volcano: :boom: ​

  It should not aim for states where it has the most control according to its current abilities, but for states where it expects it will achieve the most control after learning.

  🔹  [Ensemble and Auxiliary Tasks for Data-Efficient Deep Reinforcement Learning](https://arxiv.org/pdf/2107.01904.pdf) :no_mouth:  ​
  
   we study the effects of ensemble and auxiliary tasks when combined with the deep Q-learning alg.
  
  🔹 [Unsupervised Skill-Discovery and Skill-Learning in Minecraft](https://arxiv.org/pdf/2107.08398.pdf) :no_mouth:  ​
  
  🔹 [Variational Empowerment as Representation Learning for Goal-Based Reinforcement Learning](http://proceedings.mlr.press/v139/choi21b/choi21b.pdf) :droplet:  ​
  
  🔹 [LEARNING MORE SKILLS THROUGH OPTIMISTIC EXPLORATION](https://openreview.net/pdf?id=cU8rknuhxc) :volcano:
  
  DISDAIN (discriminator disagreement intrinsic reward): we derive an information gain auxiliary objective that involves training an ensemble of discriminators and rewarding the policy for their disagreement.
  
  🔹 [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://openreview.net/pdf?id=uqv8-U4lKBe)
  
  🔹 [LIPSCHITZ-CONSTRAINED UNSUPERVISED SKILL DISCOVERY](https://arxiv.org/pdf/2202.00914.pdf) :fire:
  
  We propose Lipschitz-constrained Skill Discovery (LSD), which encourages the agent to discover more diverse, dynamic, and far-reaching skills. LSD encourages the agent to prefer skills with larger traveled distances, unlike previous MI-based methods
  
  🔹 [THE INFORMATION GEOMETRY OF UNSUPERVISED REINFORCEMENT LEARNING](https://arxiv.org/pdf/2110.02719.pdf) :volcano:
  
  we show that unsupervised skill discovery algorithms based on MI maximization do not learn skills that are optimal for every possible reward function. However, we show that the distribution over skills provides an optimal initialization minimizing regret against adversarially-chosen reward functions, assuming a certain type of adaptation procedure.
  
  🔹 [Unsupervised Reinforcement Learning in Multiple Environments](https://arxiv.org/pdf/2112.08746.pdf) :+1:
  
  we foster an exploration strategy that is sensitive to the most adverse cases within the class. Hence, we cast the exploration problem as the maximization of the mean of a critical percentile of the state visitation entropy induced by the exploration strategy over the class of environments.  
  
<a name="anchor-inference"></a>

## Control as Inference

  🔹 [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/pdf/1805.00909.pdf) :boom: :boom: :volcano: :volcano:  ​

  **Graphic model for control as inference** (Decision Making Problem and Terminology; The Graphical Model; Policy search as Probabilistic Inference; Which Objective does This Inference Procedure Optimize;  Alternative Model Formulations);

  Variation Inference and Stochastic Dynamic(Maximium RL with Fixed Dynamics; Connection to Structured VI);

  Approximate Inference with Function Approximation(Maximum Entropy PG; Maxium Entropy AC Algorithms)

  🔹 [On Stochastic Optimal Control and Reinforcement Learning by Approximate Inference](http://www.roboticsproceedings.org/rss08/p45.pdf) :sweat_drops:  ​

  emphasizes that MaxEnt RL can be viewed as minimizing an KL divergence.

  🔹 [Iterative Inference Models](http://bayesiandeeplearning.org/2017/papers/9.pdf)  [Iterative Amortized Inference](http://proceedings.mlr.press/v80/marino18a/marino18a.pdf)  :+1: :+1:  ​ ​

  Latent Variable Models & Variational Inference &  Variational Expectation Maximization (EM) &  Inference Models

  🔹 [MAKING SENSE OF REINFORCEMENT LEARNING AND PROBABILISTIC INFERENCE](https://arxiv.org/pdf/2001.00805.pdf) :question: :sweat_drops:  ​

  🔹 [Stochastic Latent Actor-Critic: Deep Reinforcement Learning with a Latent Variable Model](https://arxiv.org/pdf/1907.00953.pdf) :volcano: :boom:  ​ ​

  The main contribution of this work is a novel and principled approach that integrates learning stochastic sequential models and RL into a single method, performing RL in the model’s learned latent space. By **formalizing the problem as a control as inference problem within a POMDP**, we show that variational inference leads to the objective of our SLAC algorithm.

  🔹 [On the Design of Variational RL Algorithms](https://joelouismarino.github.io/files/papers/2019/variational_rl/neurips_workshop_paper.pdf) :+1: :+1: :fire: ​ Good **design choices**. :sweat_drops:  :ghost:  ​

  Identify several settings that have not yet been fully explored, and we discuss general directions for improving these algorithms: VI details; (non-)Parametric; Uniform/Learned Prior.

  🔹 [VIREL: A Variational Inference Framework for Reinforcement Learning](http://papers.nips.cc/paper/8934-virel-a-variational-inference-framework-for-reinforcement-learning.pdf) :confused:  ​

  existing inference frameworks and their algorithms pose significant challenges for learning optimal policies, for example, the lack of mode capturing behaviour in pseudo-likelihood methods, difficulties learning deterministic policies in maximum entropy RL based approaches, and a lack of analysis when function approximators are used.

  🔹 [MAXIMUM A POSTERIORI POLICY OPTIMISATION](https://arxiv.org/pdf/1806.06920.pdf) :fire: :+1:  :boom:  

  MPO based on coordinate ascent on a relative entropy objective. We show that several existing methods can directly be related to our derivation.  

🔹 [V-MPO: ON-POLICY MAXIMUM A POSTERIORI POLICY OPTIMIZATION FOR DISCRETE AND CONTINUOUS CONTROL](https://arxiv.org/pdf/1909.12238.pdf) :+1: :fire: :volcano: :boom:  ​ ​ ​ ​

adapts Maximum a Posteriori Policy Optimization to the on-policy setting.

🔹 [SOFT Q-LEARNING WITH MUTUAL-INFORMATION REGULARIZATION](https://openreview.net/pdf?id=HyEtjoCqFX) :+1: :fire: :volcano:  

In this paper, we propose a theoretically motivated framework that dynamically weights the importance of actions by using the mutual information. In particular, we express the RL problem as an inference problem where the prior probability distribution over actions is subject to optimization.

 ​

-
- [Action and Perception as Divergence Minimization](https://arxiv.org/pdf/2009.01791.pdf) :boom: :volcano: ​ :ghost: [the art of design] :sweat_drops:  ​

<a name="anchor-state-abstraction"></a>

## State Abstraction, Representation Learning

  Representation learning for control based on bisimulation does not depend on reconstruction, but aims to group states based on their behavioral similarity in MDP.  [lil-log](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#bisimulation) :sweat_drops:

  🔹 Equivalence Notions and Model Minimization in Markov Decision Processes <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.2493&rep=rep1&type=pdf> : refers to an equivalence relation between two states with similar long-term behavior. :confused:  

  [BISIMULATION METRICS FOR CONTINUOUS MARKOV DECISION PROCESSES](https://www.cs.mcgill.ca/~prakash/Pubs/siamFP11.pdf)

  🔹 DeepMDP: Learning Continuous Latent Space Models for Representation Learning <https://arxiv.org/pdf/1906.02736.pdf> simplifies high-dimensional observations in RL tasks and learns a latent space model via minimizing two losses: **prediction of rewards** and **prediction of the distribution over next latent states**. :mega: :milky_way: :confused: :boom: :bomb: :boom:

🔹 [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](http://proceedings.mlr.press/v70/higgins17a/higgins17a.pdf) :fire:  

We propose a new multi-stage RL agent, DARLA (DisentAngled Representation Learning Agent), which learns to see before learning to act. DARLA’s vision is based on learning a disentangled representation of the observed environment. Once DARLA can see, it is able to acquire source policies that are robust to many domain shifts - even with no access to the target domain.



  🔹 [DBC](https://zhuanlan.zhihu.com/p/157534599): [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/pdf/2006.10742.pdf)  :boom: :boom: :boom:

  Our method trains encoders such that distances in latent space equal bisimulation distances in state space. PSM: r(s,a) ---> pi(a|s)

  🔹 [Towards Robust Bisimulation Metric Learning](https://arxiv.org/pdf/2110.14096.pdf) 🌋 :boom: 

  we generalize value function approximation bounds for on-policy bisimulation metrics to non-optimal policies and approximate environment dynamics. Our theoretical results help us identify embedding pathologies that may occur in practical use. In particular, we find that these issues stem from an underconstrained dynamics model and an unstable dependence of the embedding norm on the reward signal in environments with sparse rewards.

🔹 [TASK-INDUCED REPRESENTATION LEARNING](https://openreview.net/pdf?id=OzyXtIZAzFv) :volcano:

We formalize the problem of task-induced 11 representation learning (TARP), which aims to leverage such task information in offline experience from prior tasks for learning compact representations that focus 13 on modelling only task-relevant aspects.

🔹 [LEARNING GENERALIZABLE REPRESENTATIONS FOR REINFORCEMENT LEARNING VIA ADAPTIVE METALEARNER OF BEHAVIORAL SIMILARITIES](https://openreview.net/pdf?id=zBOI9LFpESK)  :+1: :fire:

Meta-learner of Behavioral Similarities (AMBS): A pair of meta-learners is developed, one of which quantifies the reward similarity and the other of which quantifies dynamics similarity over the correspondingly decomposed embeddings. The meta-learners are self-learned to update the state embeddings by approximating two disjoint terms in on-policy bisimulation metric.

  🔹 LEARNING INVARIANT FEATURE SPACES TO TRANSFER SKILLS WITH REINFORCEMENT LEARNING <https://arxiv.org/pdf/1703.02949.pdf> :fire: :+1:

  differ in state-space, action-space, and dynamics.

  Our method uses the skills that were learned by both agents to train **invariant feature spaces** that can then be used to transfer other skills from one agent to another.

  🔹 [UIUC: CS 598 Statistical Reinforcement Learning (S19)](http://nanjiang.cs.illinois.edu/cs598/) NanJiang :+1::eagle: :eagle:

  🔹 [CONTRASTIVE BEHAVIORAL SIMILARITY EMBEDDINGS FOR GENERALIZATION IN REINFORCEMENT LEARNING](https://openreview.net/pdf?id=qda7-sVg84) :boom: :sweat_drops:  ​

  :hourglass:  :diamond_shape_with_a_dot_inside:  :diamond_shape_with_a_dot_inside:  :hourglass: ***Rep***resentation learning. :hourglass:  :diamond_shape_with_a_dot_inside:  :diamond_shape_with_a_dot_inside:  :hourglass:

  🔹 [Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?](https://arxiv.org/pdf/2206.05266.pdf) :fire:

  we fail to find a single self-supervised loss or a combination of multiple SSL methods that consistently improve RL under the existing joint learning framework with image augmentation.

  🔹 [CURL: Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/pdf/2004.04136.pdf) :fire: :volcano: :droplet:  ​ ​

  🔹 [Denoised MDPs: Learning World Models Better Than the World Itself](https://arxiv.org/pdf/2206.15477.pdf) :fire: 🌋 

  This framework clarifies the kinds information (controllable and reward-relevant) removed by various prior work on representation learning in reinforcement learning (RL), and leads to our proposed approach of learning a Denoised MDP that explicitly factors out certain noise distractors.

🔹 [MASTERING VISUAL CONTINUOUS CONTROL: IMPROVED DATA-AUGMENTED REINFORCEMENT LEARNING](https://openreview.net/pdf?id=_SJ-_yyes8) :fire:

 DrQ-v2 builds on DrQ, an off-policy actor-critic approach that uses data augmentation to learn directly from pixels.

 🔹 [Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation](https://arxiv.org/pdf/2107.00644.pdf) :+1: 

 By only applying augmentation in Q-value estimation of the current state, without augmenting Q-targets used for bootstrapping, SVEA circumvents erroneous bootstrapping caused by data augmentation. 

  🔹 [Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks](https://arxiv.org/pdf/1812.07252.pdf) :+1: :fire:

  Our method learns to translate randomized rendered images into their equivalent non-randomized, canonical versions. This in turn allows for real images to also be translated into canonical sim images.

  🔹 [Time-contrastive networks: Self-supervised learning from video](https://arxiv.org/pdf/1704.06888.pdf) :fire:  ​

  🔹 [Data-Efficient Reinforcement Learning with Self-Predictive Representations](https://arxiv.org/pdf/2007.05929.pdf) :fire: ​  ​ ​

  SPR:

  🔹 [Value-Consistent Representation Learning for Data-Efficient Reinforcement Learning](https://arxiv.org/pdf/2206.12542.pdf) :fire: :fire:

  VCR: Instead of aligning this imagined state with a real state returned by the environment, VCR applies a Q-value head on both states and obtains two distributions of action values. Then a distance is computed and minimized to force the imagined state to produce a similar action value prediction as that by the real state.

  🔹 [Intrinsically Motivated Self-supervised Learning in Reinforcement Learning](https://arxiv.org/pdf/2106.13970.pdf) :+1: :fire:  ​

  employ self-supervised loss as an intrinsic reward, called Intrinsically Motivated Self-Supervised learning in Reinforcement learning (IM-SSR). *Decomposition and Interpretation of Contrastive Loss.*  

🔹 [INFORMATION PRIORITIZATION THROUGH EMPOWERMENT IN VISUAL MODEL-BASED RL](https://openreview.net/pdf?id=DfUjyyRW90) :+1: :fire:

InfoPower: We propose a modified objective for model-based RL that, in combination with mutual information maximization, allows us to learn representations and dynamics for visual model-based RL without reconstruction in a way that explicitly prioritizes functionally relevant factors.  

🔹 [PlayVirtual: Augmenting Cycle-Consistent Virtual Trajectories for Reinforcement Learning](https://proceedings.neurips.cc/paper/2021/file/2a38a4a9316c49e5a833517c45d31070-Paper.pdf) :+1:

PlayVirtual predicts future states in a latent space based on the current state and action by a dynamics model and then predicts the previous states by a backward dynamics model, which forms a trajectory cycle. Based on this, we augment the actions to generate a large amount of virtual state-action trajectories.

🔹 [Masked World Models for Visual Control](https://arxiv.org/pdf/2206.14244.pdf) 😶 

We train an autoencoder with convolutional layers and vision transformers (ViT) to reconstruct pixels given masked convolutional features, and learn a latent dynamics model that operates on the representations from the autoencoder. Moreover, to encode task-relevant information, we introduce an auxiliary reward prediction objective for the autoencoder.

  🔹 [EMI: Exploration with Mutual Information](https://arxiv.org/pdf/1810.01176.pdf) :+1:

  We propose EMI, which is an exploration method that constructs embedding representation of states and actions that does not rely on generative decoding of the full observation but extracts predictive signals that can be used to guide exploration based on forward prediction in the representation space.

  🔹 [Bootstrap Latent-Predictive Representations for Multitask Reinforcement Learning](https://arxiv.org/pdf/2004.14646.pdf) :+1: :fire: :volcano:

  The forward prediction encourages the agent state to move away from collapsing in order to accurately predict future random projections of observations. Similarly, the reverse prediction encourages the latent observation away from collapsing in order to accurately predict the random projection of a full history. *As we continue to train forward and reverse predictions, this seems to result in a virtuous cycle that continuously enriches both representations.*

  🔹 [Unsupervised Domain Adaptation with Shared Latent Dynamics for Reinforcement Learning](http://bayesiandeeplearning.org/2019/papers/102.pdf) :+1:

  The model achieves the alignment between the latent codes via learning shared dynamics for different environments and matching marginal distributions of latent codes.

  🔹 [RETURN-BASED CONTRASTIVE REPRESENTATION LEARNING FOR REINFORCEMENT LEARNING](https://arxiv.org/pdf/2102.10960.pdf) :+1: :fire:

  Our auxiliary loss is theoretically justified to learn representations that capture the structure of a new form of state-action abstraction, under which state-action pairs with similar return distributions are aggregated together. *Related work: AUXILIARY TASK + ABSTRACTION.*  

  🔹 [Representation Matters: Offline Pretraining for Sequential Decision Making](https://arxiv.org/pdf/2102.05815.pdf) :+1:

  🔹 [SELF-SUPERVISED POLICY ADAPTATION DURING DEPLOYMENT](https://arxiv.org/pdf/2007.04309.pdf) :+1: :fire:  ​

  test time training  [TTT](https://arxiv.org/pdf/1909.13231.pdf)         Our work explores the use of self-supervision to allow the policy to continue training after deployment without using any rewards.

  🔹 [What Makes for Good Views for Contrastive Learning?](https://arxiv.org/pdf/2005.10243.pdf) :+1:  :fire: :boom: :volcano:

  we should reduce the mutual information (MI) between views while keeping task-relevant information intact.

  🔹 [SELF-SUPERVISED LEARNING FROM A MULTI-VIEW PERSPECTIVE](https://arxiv.org/pdf/2006.05576.pdf) :+1: :fire:  ​ ​

  Demystifying Self-Supervised Learning: An Information-Theoretical Framework.

  🔹[CONTRASTIVE BEHAVIORAL SIMILARITY EMBEDDINGS FOR GENERALIZATION IN REINFORCEMENT LEARNING](https://openreview.net/pdf?id=qda7-sVg84) :+1: :+1:

  policy similarity metric (PSM) for measuring behavioral similarity between states. PSM assigns high similarity to states for which the optimal policies in those states as well as in future states are similar.

🔹 [Improving Zero-shot Generalization in Offline Reinforcement Learning using Generalized Similarity Functions](https://arxiv.org/pdf/2111.14629.pdf) :fire:

We propose a new theoretically-motivated framework called Generalized Similarity Functions (GSF), which uses contrastive learning to train an offline RL agent to aggregate observations based on the similarity of their expected future behavior, where we quantify this similarity using generalized value functions.  

  🔹 [Invariant Causal Prediction for Block MDPs](http://proceedings.mlr.press/v119/zhang20t/zhang20t.pdf) :sweat_drops:  

  State Abstractions and Bisimulation; Causal Inference Using Invariant Prediction;

🔹 [Learning Domain Invariant Representations in Goal-conditioned Block MDPs](https://arxiv.org/pdf/2110.14248.pdf)

  🔹 [CAUSAL INFERENCE Q-NETWORK: TOWARD RESILIENT REINFORCEMENT LEARNING](https://arxiv.org/pdf/2102.09677.pdf) :+1:

  In this paper, we consider a resilient DRL framework with observational interferences.

  🔹 [Decoupling Value and Policy for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2102.10330.pdf) :+1: :fire:  ​

  Invariant Decoupled Advantage ActorCritic. First, IDAAC decouples the optimization of the policy and value function, using separate networks to model them. Second, it introduces an auxiliary loss which encourages the representation to be invariant to task-irrelevant properties of the environment.

  🔹 [Robust Deep Reinforcement Learning against Adversarial Perturbations on State Obs](https://arxiv.org/pdf/2003.08938.pdf) :fire: :volcano: :droplet:  ​ ​

  We propose the state-adversarial Markov decision process (SA-MDP) to study the fundamental properties of this problem, and develop a theoretically principled policy regularization which can be applied to a large family of DRL algorithms.

  🔹 [Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning](https://arxiv.org/pdf/2106.15860.pdf) :sweat_drops:  ​

  🔹 [ROBUST REINFORCEMENT LEARNING ON STATE OBSERVATIONS WITH LEARNED OPTIMAL ADVERSARY](https://arxiv.org/pdf/2101.08452.pdf)  

  🔹 [Loss is its own Reward: Self-Supervision for Reinforcement Learning](https://arxiv.org/pdf/1612.07307.pdf)  :+1:  :boom:  ​

   To augment reward, we consider a range of selfsupervised tasks that incorporate states, actions, and successors to provide auxiliary losses.

  🔹 [Unsupervised Learning of Visual 3D Keypoints for Control](https://arxiv.org/pdf/2106.07643.pdf) :+1: :boom:

  motivation: most of these representations, whether structured or unstructured are learned in a 2D space even though the control tasks are usually performed in a 3D environment.

  🔹 [Which Mutual-Information Representation Learning Objectives are Sufficient for Control?](https://arxiv.org/pdf/2106.07278.pdf) :+1: :boom: :volcano:

  we formalize the sufficiency of a state representation for learning and representing the optimal policy, and study several popular mutual-information based objectives through this lens.  ​

  🔹 [Towards a Unified Theory of State Abstraction for MDPs](http://rbr.cs.umass.edu/aimath06/proceedings/P21.pdf) :+1: :fire::volcano: :droplet:  ​ ​ ​

  We provide a unified treatment of state abstraction for Markov decision processes. We study five particular abstraction schemes.

  🔹 [Learning State Abstractions for Transfer in Continuous Control](https://arxiv.org/pdf/2002.05518.pdf) :fire:  ​

  Our main contribution is a learning algorithm that abstracts a continuous state-space into a discrete one. We transfer this learned representation to unseen problems to enable effective learning.

  🔹 [Multi-Modal Mutual Information (MuMMI) Training for Robust Self-Supervised Deep Reinforcement Learning](https://arxiv.org/pdf/2107.02339.pdf) :+1:  :fire:  

  we contribute a new multi-modal deep latent state-space model, trained using a mutual information lower-bound.

🔹 [LEARNING ACTIONABLE REPRESENTATIONS WITH GOAL-CONDITIONED POLICIES](https://arxiv.org/pdf/1811.07819.pdf) :+1: :volcano:  ​ ​

Aim to capture those factors of variation that are important for decision making – that are “actionable.” These representations are aware of the dynamics of the environment, and capture only the elements of the observation that are necessary for decision making rather than all factors of variation.

🔹 [Adaptive Auxiliary Task Weighting for Reinforcement Learning](https://par.nsf.gov/servlets/purl/10159738) :+1:

Dynamically combines different auxiliary tasks to speed up training for reinforcement learning: Our method is based on the idea that auxiliary tasks should provide gradient directions that, in the long term, help to decrease the loss of the main task.

🔹 [Scalable methods for computing state similarity in deterministic Markov Decision Processes](https://arxiv.org/pdf/1911.09291.pdf)

Computing and approximating bisimulation metrics in large deterministic MDPs.

🔹 [Value Preserving State-Action Abstractions](http://proceedings.mlr.press/v108/abel20a/abel20a.pdf) :confused:

We proved which state-action abstractions are guaranteed to preserve representation of high value policies. To do so, we introduced -relative options, a simple but expressive formalism for combining state abstractions with options.

🔹 [Learning Markov State Abstractions for Deep Reinforcement Learning](https://arxiv.org/pdf/2106.04379.pdf) :volcano: :confused: :droplet:

 We introduce a novel set of conditions and prove that they are sufficient for learning a Markov abstract state representation. We then describe a practical training procedure that combines inverse model estimation and temporal contrastive learning to learn an abstraction that approximately satisfies these conditions.

🔹 [Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL](https://arxiv.org/pdf/2106.02193.pdf) :confused: :droplet:

We posit that a superior encoder for zero-shot generalization in RL can be trained by using solely an auxiliary SSL objective if the training process encourages the encoder to map behaviorally similar observations to similar representations.

🔹 [Jointly-Learned State-Action Embedding for Efficient Reinforcement Learning](https://arxiv.org/pdf/2010.04444.pdf) :no_mouth:

We establish the theoretical foundations for the validity of training a rl agent using embedded states and actions. We then propose a new approach for jointly learning embeddings for states and actions that combines model-free and model-based rl.

🔹 [Metrics and continuity in reinforcement learning](https://arxiv.org/pdf/2102.01514.pdf) :volcano:

We introduce a unified formalism for defining these topologies through the lens of metrics. We establish a hierarchy amongst these metrics and demonstrate their theoretical implications on the Markov Decision Process specifying the rl problem.  

🔹 [Environment Shaping in Reinforcement Learning using State Abstraction](https://arxiv.org/pdf/2006.13160.pdf) :sweat_drops: 🌋 

Our key idea is to compress the environment’s large state space with noisy signals to an abstracted space, and to use this abstraction in creating smoother and more effective feedback signals for the agent. We study the theoretical underpinnings of our abstractionbased environment shaping, and show that the agent’s policy learnt in the shaped environment preserves near-optimal behavior in the original environment.

🔹 [A RELATIONAL INTERVENTION APPROACH FOR UNSUPERVISED DYNAMICS GENERALIZATION IN MODEL-BASED REINFORCEMENT LEARNING](https://openreview.net/pdf?id=YRq0ZUnzKoZ) :fire: :volcano:

Because environments are not labelled, the extracted information inevitably contains redundant information unrelated to the dynamics in transition segments and thus fails to maintain a crucial property of Z: Z should be similar in the same environment and dissimilar in different ones. we introduce an interventional prediction module to estimate the probability of two estimated zi , zj belonging to the same environment.

🔹 [Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL](https://arxiv.org/pdf/2106.02193.pdf) :fire:

We propose Cross Trajectory Representation Learning (CTRL), a method that runs within an RL agent and conditions its encoder to recognize behavioral similarity in observations by applying a novel SSL objective to pairs of trajectories from the agent’s policies.

🔹 [Bayesian Imitation Learning for End-to-End Mobile Manipulation](https://arxiv.org/pdf/2202.07600.pdf) :+1:

We show that using the Variational Information Bottleneck to regularize convolutional neural networks improves generalization to held-out domains and reduces the sim-to-real gap in a sensor-agnostic manner. As a side effect, the learned embeddings also provide useful estimates of model uncertainty for each sensor.

🔹 [Control-Aware Representations for Model-based Reinforcement Learning](https://arxiv.org/pdf/2006.13408.pdf) :+1: :volcano: :boom: :boom:

CARL: How to learn a representation that is amenable to the control problem at hand, and how to achieve an end-to-end framework for representation learning and control: We first formulate a learning controllable embedding (LCE) model to learn representations that are suitable to be used by a policy iteration style algorithm in the latent space. We call this model control-aware representation learning (CARL). We derive a loss function for CARL that has close connection to the prediction, consistency, and curvature (PCC) principle for representation learning.

🔹 [Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images](https://arxiv.org/pdf/1506.07365.pdf) :fire:

E2C: Embed to Control (E2C) consists of a deep generative model, belonging to the family of variational autoencoders, that learns to generate image trajectories from a latent space in which the dynamics is constrained to be locally linear.

🔹 [Robust Locally-Linear Controllable Embedding](http://proceedings.mlr.press/v84/banijamali18a/banijamali18a.pdf) :fire:  

RCE: propose a principled variational approximation of the embedding posterior that takes the future observation into account, and thus, makes the variational approximation more robust against the noise.

🔹 [SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning](https://arxiv.org/pdf/1808.09105.pdf) :no_mouth:

SOLAR: we present a method for learning representations that are suitable for iterative model-based policy improvement.

🔹 [DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION](https://arxiv.org/pdf/1912.01603.pdf) :+1:

Dreamer: (Learning long-horizon behaviors by latent imagination) predicting *both actions and state values*.  

🔹 [Learning Task Informed Abstractions](http://proceedings.mlr.press/v139/fu21b/fu21b.pdf) 😶 

Task Informed Abstractions (TIA) that explicitly separates rewardcorrelated visual features from distractors.

🔹 [PREDICTION, CONSISTENCY, CURVATURE: REPRESENTATION LEARNING FOR LOCALLY-LINEAR CONTROL](https://openreview.net/pdf?id=BJxG_0EtDS) :+1: :fire: :volcano:

PCC: We propose the Prediction, Consistency, and Curvature (PCC) framework for learning a latent space that is amenable to locally-linear control (LLC) algorithms and show that the elements of PCC arise systematically from bounding the suboptimality of the solution of the LLC algorithm in the latent space.

🔹 [Predictive Coding for Locally-Linear Control](https://arxiv.org/pdf/2003.01086.pdf) :fire: :volcano:

PC3: we propose a novel information-theoretic LCE approach and show theoretically that explicit next-observation prediction can be replaced with predictive coding. We then use predictive coding to develop a decoder-free LCE model whose latent dynamics are amenable to locally-linear control.

🔹 [Robust Predictable Control](https://arxiv.org/pdf/2109.03214.pdf) :fire: :volcano: :droplet:

RPC: Our objective differs from prior work by compressing sequences of observations, resulting in a method that jointly trains a policy and a model to be self-consistent.

🔹 [Representation Gap in Deep Reinforcement Learning](https://arxiv.org/pdf/2205.14557.pdf) 🌋 

We propose Policy Optimization from Preventing Representation Overlaps (POPRO), which regularizes the policy evaluation phase through differing the representation of action value function from its target. 

🔹 [TRANSFER RL ACROSS OBSERVATION FEATURE SPACES VIA MODEL-BASED REGULARIZATION](https://arxiv.org/pdf/2201.00248.pdf) :fire: :fire: 

We propose to learn a latent dynamics model in the source task and transfer the model to the target task to facilitate representation learning (+heoretical analysis). 

🔹 [Sample-Efficient Reinforcement Learning in the Presence of Exogenous Information](https://arxiv.org/pdf/2206.04282.pdf) :fire:

ExoMDP: the state space admits an (unknown) factorization into a small controllable (or, endogenous) component and a large irrelevant (or, exogenous) component; the exogenous component is independent of the learner’s actions, but evolves in an arbitrary,
temporally correlated fashion.

🔹 [Stabilizing Off-Policy Deep Reinforcement Learning from Pixels](https://arxiv.org/pdf/2207.00986.pdf) 🌋 

A-LIX: [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/caf1a3dfb505ffed0d024130f58c5cfa_BpJWwwI.png) 

🔹 [Temporal Disentanglement of Representations for Improved Generalisation in Reinforcement Learning](https://arxiv.org/pdf/2207.05480.pdf) :+1: 

we introduce TEmporal Disentanglement (TED), a self-supervised auxiliary task that leads to disentangled representations using the sequential nature of RL observations.

🔹 [R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/pdf/2203.12601.pdf) 

We study how visual representations pre-trained on diverse human video data can enable data-efficient learning of downstream robotic manipulation tasks.




<a name="anchor-MI"></a>

## Mutual Information  

  🔹 [MINE: Mutual Information Neural Estimation](https://arxiv.org/pdf/1801.04062.pdf) :+1::droplet:  :fire:     [f-gan & mine](https://zhuanlan.zhihu.com/p/151256189) :sweat_drops:

🔹 [IMPROVING MUTUAL INFORMATION ESTIMATION WITH ANNEALED AND ENERGY-BASED BOUNDS](https://openreview.net/pdf?id=T0B9AoM_bFg)

Multi-Sample Annealed Importance Sampling (AIS):

  🔹 [C-MI-GAN : Estimation of Conditional Mutual Information Using MinMax Formulation](https://arxiv.org/pdf/2005.08226.pdf) :+1: :fire:  ​ ​

  🔹 [Deep InfoMax: LEARNING DEEP REPRESENTATIONS BY MUTUAL INFORMATION ESTIMATION AND MAXIMIZATION](https://arxiv.org/pdf/1808.06670.pdf) :+1::droplet: ​  

  🔹 [ON MUTUAL INFORMATION MAXIMIZATION FOR REPRESENTATION LEARNING](https://arxiv.org/pdf/1907.13625.pdf) :sweat_drops: :+1:  ​

  🔹 [Deep Reinforcement and InfoMax Learning](Deep Reinforcement and InfoMax Learning) :sweat_drops: :+1: :confused: :droplet:

  Our work is based on the hypothesis that a model-free agent whose **representations are predictive of properties of future states** (beyond expected rewards) will be more capable of solving and adapting to new RL problems, and in a way, incorporate aspects of model-based learning.

  🔹 [Unpacking Information Bottlenecks: Unifying Information-Theoretic Objectives in Deep Learning](https://arxiv.org/pdf/2003.12537.pdf) :volcano:

  New: Unpacking Information Bottlenecks: Surrogate Objectives for Deep Learning

  🔹 [Opening the black box of Deep Neural Networ ksvia Information](https://arxiv.org/pdf/1703.00810.pdf)

  :o: :o: UYANG:

  [小王爱迁移](https://zhuanlan.zhihu.com/p/27336930),

  🔹 ​[Self-Supervised Representation Learning From Multi-Domain Data](Self-Supervised Representation Learning From Multi-Domain Data), :fire: :+1: :+1:

  The proposed mutual information constraints encourage neural network to extract common invariant information across domains and to preserve peculiar information of each domain simultaneously. We adopt tractable **upper and lower bounds of mutual information** to make the proposed constraints solvable.

  🔹 ​[Unsupervised Domain Adaptation via Regularized Conditional Alignment](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cicek_Unsupervised_Domain_Adaptation_via_Regularized_Conditional_Alignment_ICCV_2019_paper.pdf), :fire: :+1:  

   Joint alignment ensures that not only the marginal distributions of the domains are aligned, but the labels as well.

  🔹 ​[Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift](https://arxiv.org/pdf/2003.04475.pdf), :fire: :boom: :boom: :sweat_drops:

  In this paper, we extend a recent upper-bound on the performance of adversarial domain adaptation to multi-class classification and more general discriminators. We then propose **generalized label shift (GLS)** as a way to improve robustness against mismatched label distributions. GLS states that, conditioned on the label, **there exists a representation of the input that is invariant between the source and target domains**.

  🔹 ​[Learning to Learn with Variational Information Bottleneck for Domain Generalization](https://arxiv.org/pdf/2007.07645.pdf),

  Through episodic training, MetaVIB learns to gradually narrow domain gaps to establish domain-invariant representations, while simultaneously maximizing prediction accuracy.

  🔹 ​[Deep Domain Generalization via Conditional Invariant Adversarial Networks](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf), :+1:

  🔹 ​[On Learning Invariant Representation for Domain Adaptation](https://arxiv.org/pdf/1901.09453.pdf) :fire: :boom: :sweat_drops:

  🔹 ​[GENERALIZING ACROSS DOMAINS VIA CROSS-GRADIENT TRAINING](https://arxiv.org/pdf/1804.10745.pdf) :fire: :volcano: :volcano:

  In contrast, in our setting, we wish to avoid any such explicit domain representation, **appealing instead to the power of deep networks to discover implicit features**. We also argue that even if such such overfitting could be avoided, we do not necessarily want to wipe out domain signals, if it helps in-domain test instances.

  🔹 ​[In Search of Lost Domain Generalization](https://arxiv.org/pdf/2007.01434.pdf) :no_mouth:  

  🔹 [DIRL: Domain-Invariant Representation Learning for Sim-to-Real Transfer](https://arxiv.org/pdf/2011.07589.pdf) :sweat_drops:  ​

  :o: :o: ***self-supervised*** learning

  🔹 [Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf) :fire: :volcano:

  Related work is good!  ​ ​

🔹 [Model-Based Relative Entropy Stochastic Search](http://proceedings.mlr.press/v80/arenz18a/arenz18a.pdf)  

MORE:

🔹 [Efficient Gradient-Free Variational Inference using Policy Search](http://proceedings.mlr.press/v80/arenz18a/arenz18a.pdf)  

VIPS: Our method establishes information-geometric trust regions to ensure efficient exploration of the sampling space and stability of the GMM updates, allowing for efficient estimation of multi-variate Gaussian variational distributions.

🔹 [EXPECTED INFORMATION MAXIMIZATION USING THE I-PROJECTION FOR MIXTURE DENSITY ESTIMATION](https://arxiv.org/pdf/2001.08682.pdf) :fire:

EIM: we present a new algorithm called Expected Information Maximization (EIM) for computing the I-projection solely based on samples for general latent variable models.

🔹 [An Information-theoretic Approach to Distribution Shifts](https://openreview.net/pdf?id=GrZmKDYCp6H) :droplet:

🔹 [An Asymmetric Contrastive Loss for Handling Imbalanced Datasets](https://arxiv.org/pdf/2207.07080.pdf) :fire: 🌋 

we propose the asymmetric focal contrastive loss (AFCL) as a further generalization of both ACL and focal contrastive loss (FCL). 

<a name="anchor-DR"></a>  <a name="anchor-sim2real"></a>  

## DR (Domain Randomization) & sim2real

  🔹 Active Domain Randomization <http://proceedings.mlr.press/v100/mehta20a/mehta20a.pdf> :fire: :boom: :fire:

  Our method looks for the most **informative environment variations** within the given randomization ranges by **leveraging the discrepancies of policy rollouts in randomized and reference environment instances**. We find that training more frequently on these instances leads to better overall agent generalization.

  Domain Randomization; Stein Variational Policy Gradient;

  Bhairav Mehta [On Learning and Generalization in Unstructured Task Spaces](https://bhairavmehta95.github.io/static/thesis.pdf) :sweat_drops: :sweat_drops:

  🔹 [VADRA: Visual Adversarial Domain Randomization and Augmentation](https://arxiv.org/pdf/1812.00491.pdf) :fire: :+1:  generative + learner

  🔹 [Which Training Methods for GANs do actually Converge?](https://arxiv.org/pdf/1801.04406.pdf)  :+1: :droplet:   [ODE: GAN](https://zhuanlan.zhihu.com/p/65953336)

  🔹 [Robust Adversarial Reinforcement Learning](https://arxiv.org/pdf/1703.02702.pdf) :no_mouth:  ​

  Robust Adversarial Reinforcement Learning (RARL), jointly trains a pair of agents, a protagonist and an adversary, where the protagonist learns to fulfil the original task goals while being robust to the disruptions generated by its adversary.

  🔹 [Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience](https://arxiv.org/pdf/1810.05687.pdf) :no_mouth:  ​

  🔹 [POLICY TRANSFER WITH STRATEGY OPTIMIZATION](https://arxiv.org/pdf/1810.05751.pdf) :no_mouth:  ​

  The key idea that, instead of learning a single policy in the simulation, we simultaneously learn a family of policies that exhibit different behaviors. When tested in the target environment, we directly search for the best policy in the family based on the task performance, without the need to identify the dynamic parameters.

  🔹 <https://lilianweng.github.io/lil-log/2019/05/05/domain-randomization.html> :sweat_drops:

  🔹 [THE INGREDIENTS OF REAL-WORLD ROBOTIC REINFORCEMENT LEARNING](https://arxiv.org/pdf/2004.12570.pdf) :no_mouth:  ​

  🔹 [ROBUST REINFORCEMENT LEARNING ON STATE OBSERVATIONS WITH LEARNED OPTIMAL ADVERSARY](https://openreview.net/pdf?id=sCZbhBvqQaU) :+1:  

  To enhance the robustness of an agent, we propose a framework of alternating training with learned adversaries (ATLA), which trains an adversary online together with the agent using policy gradient following the optimal adversarial attack framework.  

  🔹 [SELF-SUPERVISED POLICY ADAPTATION DURING DEPLOYMENT](https://openreview.net/pdf?id=o_V-MjyyGV_) :+1: :fire:  

  Our work explores the use of self-supervision to allow the policy to continue training after deployment without using any rewards.

  🔹 [SimGAN: Hybrid Simulator Identification for Domain Adaptation via Adversarial Reinforcement Learning](https://arxiv.org/pdf/2101.06005.pdf) :+1:

  identifying a hybrid physics simulator to match the simulated \tau to the ones from the target domain, using a learned discriminative loss to address the limitations associated with manual loss design. Our hybrid simulator combines nns and traditional physics simulaton to balance expressiveness and generalizability, and alleviates the need for a carefully selected parameter set in System ID.

🔹 [Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation](https://arxiv.org/pdf/2106.15587.pdf) :no_mouth:

our proposed method adversarially generates new trajectory data based on the policy gradient objective and aims to more effectively increase the RL agent’s generalization ability with the policy-aware data augmentation.

🔹 [Understanding Domain Randomization for Sim-to-real Transfer](https://openreview.net/pdf?id=T8vZHIRTrY) :volcano: :droplet:

We provide sharp bounds on the sim-to-real gap—the difference between the value of policy returned by domain randomization and the value of an optimal policy for the real world.

:white_flag:  see more robustness in model-based setting

🔹 [EPOPT: LEARNING ROBUST NEURAL NETWORK POLICIES USING MODEL ENSEMBLES](https://arxiv.org/pdf/1610.01283.pdf) :no_mouth:  

Our method provides for training of robust policies, and supports an adversarial training regime designed to provide good direct-transfer performance. We also describe how our approach can be combined with Bayesian model adaptation to adapt the source domain ensemble to a target domain using a small amount of target domain experience.

🔹 [Action Robust Reinforcement Learning and Applications in Continuous Control](https://arxiv.org/pdf/1901.09184.pdf) :fire: :droplet:

We have presented two new criteria for robustness, the Probabilistic and Noisy action Robust MDP, related each to real world scenarios of uncertainty and discussed the theoretical differences between both approaches.  

🔹 [Robust Policy Learning over Multiple Uncertainty Sets](https://arxiv.org/pdf/2202.07013.pdf) :droplet:

 System Identification and Risk-Sensitive Adaptation (SIRSA):  

🔹 [∇Sim: DIFFERENTIABLE SIMULATION FOR SYSTEM IDENTIFICATION AND VISUOMOTOR CONTROL](https://arxiv.org/pdf/2104.02646.pdf) :+1: :droplet:

🔹 [RISP: RENDERING-INVARIANT STATE PREDICTOR WITH DIFFERENTIABLE SIMULATION AND RENDERING FOR CROSS-DOMAIN PARAMETER ESTIMATION](https://openreview.net/forum?id=uSE03demja) :fire: :volcano:  :+1:

This work considers identifying parameters characterizing a physical system’s dynamic motion directly from a video whose rendering configurations are inaccessible. Our core idea is to train a rendering-invariant state-prediction (RISP) network that transforms image differences into state differences independent of rendering configurations.

🔹 [Sim and Real: Better Together](https://openreview.net/pdf?id=t0B9XQwRDi) :fire: :droplet:

By separating the rate of collecting samples from each environment and the rate of choosing samples for the optimization process, we were able to achieve a significant reduction in the amount of real environment samples, comparing to the common strategy of using the same rate for both collection and optimization phases.

🔹 [Online Robust Reinforcement Learning with Model Uncertainty](https://openreview.net/pdf?id=IhiU6AJYpDs) :volcano:

We develop a sample-based approach to estimate the unknown uncertainty set, and design robust Q-learning algorithm (tabular case) and robust TDC algorithm (function approximation setting).

🔹 [Robust Deep Reinforcement Learning through Adversarial Loss](https://openreview.net/pdf?id=eaAM_bdW0Q) :fire: :fire:

RADIAL-RL: Construct an strict upper bound of the perturbed standard loss; Design a regularizer to minimize overlap between output bounds of actions with large difference in outcome.

🔹 [Robust Deep Reinforcement Learning through Bootstrapped Opportunistic Curriculum](https://arxiv.org/pdf/2206.10057.pdf) 💧 

🔹 [Robust Reinforcement Learning using Offline Data](https://arxiv.org/pdf/2208.05129.pdf) :fire: 👍

This poses challenges in offline data collection, optimization over the models, and unbiased estimation. In this work, we propose a systematic approach to overcome these challenges, resulting in our RFQI algorithm.
 
 ​

<a name="anchor-transfer"></a>  

## Transfer: Generalization & Adaption (Dynamics)

  🔹 [Automatic Data Augmentation for Generalization in Deep Reinforcement Learning](https://arxiv.org/pdf/2006.12862.pdf)  :punch: :+1:  ​  ​ ​

  Across different visual inputs (with the same semantics), dynamics, or other environment structures

  🔹 [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/pdf/2004.13649.pdf) :+1:

  🔹 [Fast Adaptation to New Environments via Policy-Dynamics Value Functions](https://arxiv.org/pdf/2007.02879.pdf) :fire: :boom: :+1:  ​

  PD-VF explicitly estimates the cumulative reward in a space of policies and environments.

  🔹 [Off-Dynamics Reinforcement Learning: Training for Transfer with Domain Classifiers](https://openreview.net/pdf?id=eqBwg3AcIAK) :fire: :boom: :volcano: :droplet:  

  DARC: The main contribution of this work is an algorithm for domain adaptation to dynamics changes in RL, based on the idea of compensating for **differences in dynamics** by modifying the reward function. This algorithm does not need to estimate transition probabilities, but rather modifies the reward function using a pair of classifiers.

  🔹 [DARA: DYNAMICS-AWARE REWARD AUGMENTATION IN OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2203.06662.pdf) :fire:

  🔹 [When to Trust Your Simulator: Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning](https://arxiv.org/pdf/2206.13464.pdf) :fire: 

  H2O introduces a dynamics-aware policy evaluation scheme, which adaptively penalizes the Q function learning on simulated stateaction pairs with large dynamics gaps, while also simultaneously allowing learning from a fixed real-world dataset.

  **Related work is good!** :+1:

- general domain adaption (DA) =  importance weighting + domain-agnostic features

- DA in RL =  system identification + domain randomization + observation adaptation :+1:
  - formulates control as a problem of probabilistic inference :droplet:

🔹 [Unsupervised Domain Adaptation with Dynamics Aware Rewards in Reinforcement Learning](https://arxiv.org/pdf/2110.12997.pdf) :+1: :fire: :volcano:

DADS: we introduce a KL regularized objective to encourage emergence of skills, rewarding the agent for both discovering skills and aligning its behaviors respecting dynamics shifts.

🔹 [Mutual Alignment Transfer Learning](https://arxiv.org/pdf/1707.07907.pdf) :+1: :fire:  ​ ​

The developed approach harnesses auxiliary rewards to guide the exploration for the real world agent based on the proficiency of the agent in simulation and vice versa.

🔹 [SimGAN: Hybrid Simulator Identification for Domain Adaptation via Adversarial Reinforcement Learning](https://arxiv.org/pdf/2101.06005.pdf)  [real dog] :+1: :fire:  ​ ​

a framework to tackle domain adaptation by identifying a hybrid physics simulator to match the simulated trajectories to the ones from the target domain, using a learned discriminative loss to address the limitations associated with manual loss design.

 🔹 [Disentangled Skill Embeddings for Reinforcement Learning](https://arxiv.org/pdf/1906.09223.pdf) :fire: :volcano: :boom: :boom:  ​ ​ ​ ​

We have developed a multi-task framework from a variational inference perspective that is able to learn latent spaces that generalize to unseen tasks where the dynamics and reward can change independently.

🔹 [**Transfer Learning in Deep Reinforcement Learning: A Survey**](https://arxiv.org/pdf/2009.07888.pdf)  :sweat_drops:  ​

Evaluation metrics: Mastery and Generalization.

TRANSFER LEARNING APPROACHES: Reward Shaping; Learning from Demonstrations; Policy Transfer (Transfer Learning via Policy Distillation, Transfer Learning via Policy Reuse); Inter-Task Mapping; Representation Transfer(Reusing Representations, Disentangling Representations);

🔹 [Provably Efficient Model-based Policy Adaptation](https://arxiv.org/pdf/2006.08051.pdf) :+1: :fire: :volcano: :droplet: :+1:  ​

We prove that the approach learns policies in the target environment that can recover trajectories from the source environment, and establish the rate of convergence in general settings.

:o: reward shaping

- 🔹 [Useful Policy Invariant Shaping from Arbitrary Advice](https://arxiv.org/pdf/2011.01297.pdf) :+1:  ​

- Action
  
🔹 [Generalization to New Actions in Reinforcement Learning](https://arxiv.org/pdf/2011.01928.pdf) :+1:

  We propose a two-stage framework where the agent first infers action representations from action information acquired separately from the task. A policy flexible to varying action sets is then trained with generalization objectives.

🔹 [Policy Transfer across Visual and Dynamics Domain Gaps via Iterative Grounding](https://arxiv.org/pdf/2107.00339.pdf) :+1: :fire:  

alternates between (1) directly minimizing both visual and dynamics domain gaps by grounding the source env in the target env domains, and (2) training a policy on the grounded source env.

🔹 [Learning Agile Robotic Locomotion Skills by Imitating Animals](https://arxiv.org/pdf/2004.00784.pdf) :+1: :fire: :volcano:  ​

We show that by leveraging reference motion data, a single learning-based approach is able to automatically synthesize controllers for a diverse repertoire behaviors for legged robots. By incorporating sample efficient domain adaptation techniques into the training process, our system is able to learn adaptive policies in simulation that can then be quickly adapted for real-world deployment.

🔹 [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/pdf/2107.04034.pdf) :+1:  

The robot achieves this high success rate despite never having seen unstable or sinking ground, obstructive vegetation or stairs during training. All deployment results are with the same policy without any simulation calibration, or real-world fine-tuning.  ​ ​ 

🔹 [A System for General In-Hand Object Re-Orientation](https://openreview.net/pdf?id=7uSBJDoP7tY) 👍 

We present a simple model-free framework (teacher-student distillation) that can learn to reorient objects with both the hand facing upwards and downwards.​ + DAgger

🔹 [LEARNING VISION-GUIDED QUADRUPEDAL LOCOMOTION END-TO-END WITH CROSS-MODAL TRANSFORMERS](https://arxiv.org/pdf/2107.03996.pdf) :fire:

LocoTransformer: We propose to address quadrupedal locomotion tasks using Reinforcement Learning (RL) with a Transformer-based model that learns to combine proprioceptive information and high-dimensional depth sensor inputs.  

🔹 [Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability](https://arxiv.org/pdf/2107.06277.pdf) :fire:  

we recast the problem of generalization in RL as solving the induced partially observed Markov decision process, which we call the epistemic POMDP.

🔹 [Learning quadrupedal locomotion over challenging terrain](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/448343/1/2020_science_robotics_lee_locomotion.pdf) :fire: :volcano:

 We present a novel solution to incorporating proprioceptive feedback in locomotion control and demonstrate remarkable zero-shot generalization from simulation to natural environments.

🔹 [Rma: Rapid motor adaptation for legged robots](https://arxiv.org/pdf/2107.04034.pdf) :fire: :volcano:

RMA consists of two components: a base policy and an adaptation module. The combination of these components enables the robot to adapt to novel situations in fractions of a second. RMA is trained completely in simulation without using any domain knowledge like reference trajectories or predefined foot trajectory generators and is deployed on the A1 robot without any fine-tuning.  

🔹 [Fast Adaptation to New Environments via Policy-Dynamics Value Functions](https://arxiv.org/pdf/2007.02879.pdf) :fire: 

PD-VF:  explicitly estimates the cumulative reward in a space of policies and environments.

🔹 [PAnDR: Fast Adaptation to New Environments from Offline Experiences via Decoupling Policy and Environment Representations](https://arxiv.org/pdf/2204.02877.pdf) :fire: 👍 

In offline phase, the environment representation and policy representation are learned
through contrastive learning and policy recovery, respectively. The representations are further refined by mutual information optimization to make them more decoupled and complete.

🔹 [Learning Robust Policy against Disturbance in Transition Dynamics via State-Conservative Policy Optimization](https://arxiv.org/pdf/2112.10513.pdf) :volcano:

 State-Conservative Policy Optimization (SCPO) reduces the disturbance in transition dynamics to that in state space and then approximates it by a simple gradient-based regularizer.

🔹 [LEARNING A SUBSPACE OF POLICIES FOR ONLINE ADAPTATION IN REINFORCEMENT LEARNING](https://openreview.net/pdf?id=4Muj-t_4o4) :no_mouth:

 LoP does not need any particular tuning or definition of additional architectures to handle diversity, which is a critical aspect in the online adaptation setting where hyper-parameters tuning is impossible or at least very difficult.

🔹 [ADAPT-TO-LEARN: POLICY TRANSFER IN REINFORCEMENT LEARNING](https://openreview.net/pdf?id=ryeT10VKDH) :+1: :+1:  ​

New: [Adaptive Policy Transfer in Reinforcement Learning](https://arxiv.org/pdf/2105.04699.pdf)

adapt the source policy to learn to solve a target task with significant **transition differences** and uncertainties.  

🔹 [Unsupervised Domain Adaptation with Dynamics Aware Rewards in Reinforcement Learning](https://arxiv.org/pdf/2110.12997.pdf) :fire: :volcano:  

DARS:  We propose an unsupervised domain adaptation method to identify and acquire skills across dynamics. We introduce a KL regularized objective to encourage emergence of skills, rewarding the agent for both discovering skills and aligning its behaviors respecting dynamics shifts.  

🔹 [SINGLE EPISODE POLICY TRANSFER IN REINFORCEMENT LEARNING](https://arxiv.org/pdf/1910.07719.pdf) :fire: :+1:  ​ ​

Our key idea of optimized probing for accelerated latent variable inference is to train a dedicated probe policy πϕ(a|s) to generate a dataset D of short trajectories at the beginning of all training episodes, such that the VAE’s performance on D is optimized.  

🔹 [Dynamical Variational Autoencoders: A Comprehensive Review](https://arxiv.org/pdf/2008.12595.pdf) :sweat_drops: :sweat_drops:  ​ ​

🔹 [Dynamics Generalization via Information Bottleneck in Deep Reinforcement Learning](https://arxiv.org/pdf/2008.00614.pdf)​ :fire:  ​ ​

In particular, we show that the poor generalization in unseen tasks is due to the **DNNs memorizing environment observations**, rather than extracting the relevant information for a task. To prevent this, we impose communication constraints as an information bottleneck between the agent and the environment.

🔹 [UNIVERSAL AGENT FOR DISENTANGLING ENVIRONMENTS AND TASKS](https://openreview.net/pdf?id=B1mvVm-C-) :fire: :volcano:  

The environment-specific unit handles how to move from one state to the target state; and the task-specific unit plans for the next target state given a specific task.  

🔹 [Decoupling Dynamics and Reward for Transfer Learning](https://arxiv.org/pdf/1804.10689.pdf) :+1:  

We separate learning the task representation, the forward dynamics, the inverse dynamics and the reward function of the domain.  

🔹 [Neural Dynamic Policies for End-to-End Sensorimotor Learning](https://biases-invariances-generalization.github.io/pdf/big_15.pdf) :fire: :volcano:  

We propose Neural Dynamic Policies (NDPs) that make predictions in trajectory distribution space as opposed to raw control spaces. [see Abstract!] **Similar in spirit to UNIVERSAL AGENT.**  

🔹 [Accelerating Reinforcement Learning with Learned Skill Priors](https://arxiv.org/pdf/2010.11944.pdf) :+1: :fire:  

We propose a deep latent variable model that jointly learns an embedding space of skills and the skill prior from offline agent experience. We then extend common maximumentropy RL approaches to use **skill priors to guide downstream learning**.

🔹 [Mutual Alignment Transfer Learning](https://arxiv.org/pdf/1707.07907.pdf) :+1: :fire:

The developed approach harnesses auxiliary rewards to guide the exploration for the real world agent based on the proficiency of the agent in simulation and vice versa.

🔹 [LEARNING CROSS-DOMAIN CORRESPONDENCE FOR CONTROL WITH DYNAMICS CYCLE-CONSISTENCY](https://openreview.net/pdf?id=QIRlze3I6hX) :+1: :+1: :fire:

In this paper, we propose to learn correspondence across such domains emphasizing on differing modalities (vision and internal state), physics parameters (mass and friction), and morphologies (number of limbs). Importantly, correspondences are learned using unpaired and randomly collected data from the two domains. We propose **dynamics cycles** that align dynamic robotic behavior across two domains using a cycle consistency constraint.  

🔹 [Hierarchically Decoupled Imitation for Morphological Transfer](http://proceedings.mlr.press/v119/hejna20a/hejna20a.pdf) 😶 

 incentivizing a complex agent’s low-level to imitate a simpler agent’s low-level significantly improves zero-shot high-level transfer; KL-regularized training of the high level stabilizes learning and prevents modecollapse.

🔹 [Improving Generalization in Reinforcement Learning with Mixture Regularization](https://arxiv.org/pdf/2010.10814.pdf) :+1:  

these approaches only locally perturb the observations regardless of the training environments, showing limited effectiveness on enhancing the data diversity and the generalization performance.

🔹 [AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning](https://arxiv.org/pdf/2107.02729.pdf) :fire:  :droplet:  

 we characterize a minimal set of representations, including both domain-specific factors and domain-shared state representations, that suffice for reliable and low-cost transfer.  

🔹 [A GENERAL THEORY OF RELATIVITY IN REINFORCEMENT LEARNING](https://openreview.net/pdf?id=bi9j5yi-Vrv) :fire: :volcano:

The proposed theory deeply investigates the connection between any two cumulative expected returns defined on different policies and environment dynamics: Relative Policy Optimization (RPO) updates the policy using the relative policy gradient to transfer the policy evaluated in one environment to maximize the return in another, while Relative Transition Optimization (RTO) updates the parameterized dynamics model (if there exists) using the relative transition gradient to reduce the gap between the dynamics of the two environments.  

🔹 [COPA: CERTIFYING ROBUST POLICIES FOR OFFLINE REINFORCEMENT LEARNING AGAINST POISONING ATTACKS](https://openreview.net/pdf?id=psh0oeMSBiF) :+1:

We focus on certifying the robustness of offline RL in the presence of poisoning attacks, where a subset of training trajectories could be arbitrarily manipulated. We propose the first certification framework, COPA to certify the number of poisoning trajectories that can be tolerated regarding different certification criteria.  

🔹 [CROP: CERTIFYING ROBUST POLICIES FOR REINFORCEMENT LEARNING THROUGH FUNCTIONAL SMOOTHING](https://openreview.net/pdf?id=HOjLHrlZhmx) :fire:

We propose two particular types of robustness certification criteria: robustness of per-state actions and lower bound of cumulative rewards.

🔹 [Learning Action Translator for Meta Reinforcement Learning on Sparse-Reward Tasks](https://arxiv.org/pdf/2207.09071.pdf) :fire: :fire: 

MCAT: we propose to learn an action translator among multiple training tasks. The objective function forces the translated action to behave on the target task similarly to the source action on the source task. We consider the policy transfer for any pair of source and target tasks in the training task distribution.

🔹 [AACC: Asymmetric Actor-Critic in Contextual Reinforcement Learning](https://arxiv.org/pdf/2208.02376.pdf) :fire: 

the critic is trained with environmental factors and observation while the actor only gets the observation as input.

:o: Multi-task

🔹 [Multi-Task Reinforcement Learning without Interference](https://www.skillsworkshop.ai/uploads/1/2/1/5/121527312/multi-task.pdf) :fire:

We develop a general approach that can change the multi-task optimization landscape to alleviate conflicting gradients across tasks, one architectural and one algorithmic, that prevent gradients for different tasks from interfering with one another.

🔹 [Multi-Task Reinforcement Learning with Soft Modularization](https://proceedings.neurips.cc/paper/2020/file/32cfdce9631d8c7906e8e9d6e68b514b-Paper.pdf) :no_mouth:

Given a base policy network, we design a routing network which estimates different routing strategies to reconfigure the base network for each task.  

🔹 [Multi-task Batch Reinforcement Learning with Metric Learning](https://proceedings.neurips.cc/paper/2020/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf) :fire:

MBML: Because the different datasets may have state-action distributions with large divergence, the task inference module can learn to ignore the rewards and spuriously correlate only state-action pairs to the task identity, leading to poor test time performance. To robustify task inference, we propose a novel application of the triplet loss.  

🔹 [MULTI-BATCH REINFORCEMENT LEARNING VIA SAMPLE TRANSFER AND IMITATION LEARNING](https://openreview.net/pdf?id=KTF1h2XWKZA) :no_mouth:  

BAIL+ and MBAIL  

🔹 [Knowledge Transfer in Multi-Task Deep Reinforcement Learning for Continuous Control](https://proceedings.neurips.cc/paper/2020/file/acab0116c354964a558e65bdd07ff047-Paper.pdf) :no_mouth:

KTM-DRL enables a single multi-task agent to leverage the offline knowledge transfer, the online learning, and the hierarchical experience replay for achieving expert-level performance in multiple different continuous control tasks.  

🔹 [Multi-Task Reinforcement Learning with Context-based Representations](http://proceedings.mlr.press/v139/sodhani21a/sodhani21a.pdf) :fire:

CARE: We posit that an efficient approach to knowledge transfer is through the use of multiple context-dependent, composable representations shared across a family of tasks. Metadata can help to learn interpretable representations and provide the context to inform which representations to compose and how to compose them.  

🔹 [CARL: A Benchmark for Contextual and Adaptive Reinforcement Learning](https://arxiv.org/pdf/2110.02102.pdf) :+1:  

We propose CARL, a collection of well-known RL environments extended to contextual RL problems to study generalization.

🔹 [Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning](https://arxiv.org/pdf/2203.07413.pdf) :no_mouth:  

We propose SwitchTT, a multi-task extension to Trajectory Transformer but enhanced with two striking features: (i) exploiting a sparsely activated model to reduce computation cost in multitask offline model learning and (ii) adopting a distributional trajectory value estimator that improves policy performance, especially in sparse reward settings.  

🔹 [MULTI-CRITIC ACTOR LEARNING: TEACHING RL POLICIES TO ACT WITH STYLE](https://openreview.net/pdf?id=rJvY_5OzoI) :+1:  

Multi-Critic Actor Learning (MultiCriticAL) proposes instead maintaining separate critics for each task being trained while training a single multi-task actor.  

🔹 [Investigating Generalisation in Continuous Deep Reinforcement Learning](https://arxiv.org/pdf/1902.07015.pdf)

🔹 [Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots](https://papers.nips.cc/paper/2021/file/118921efba23fc329e6560b27861f0c2-Paper.pdf) :fire:  

🔹 [Beyond Tabula Rasa: Reincarnating Reinforcement Learning](https://arxiv.org/pdf/2206.01626.pdf) 🌋 

As a step towards enabling reincarnating RL from any agent to any other agent, we focus on the specific setting of efficiently transferring an existing sub-optimal policy to a standalone valuebased RL agent. 

🔹 [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/pdf/1011.0686.pdf) :+1: :fire: 

DAgger (Dataset Aggregation): trains a deterministic policy that achieves good performance guarantees under its induced distribution of states.

🔹 [Multifidelity Reinforcement Learning with Control Variates](https://arxiv.org/pdf/2206.05165.pdf) :fire:

MFMCRL: a multifidelity estimator that exploits the cross-correlations between the low- and high-fidelity returns is proposed to reduce the variance in the estimation of the state-action value function.

🔹 [Robust Trajectory Prediction against Adversarial Attacks](https://arxiv.org/pdf/2208.00094.pdf) :+1: 

we propose an adversarial training framework with three main components, including (1) a deterministic attack for the inner maximization process of the adversarial training, (2) additional regularization terms for stable outer minimization of adversarial training, and (3) a domain-specific augmentation strategy to achieve a better performance trade-off on clean and adversarial data.



 <a name="anchor-ood"></a>

:o: :o: :o: **Out-of-Distribution (OOD) Generalization**  [Modularity--->Generalization](https://zhuanlan.zhihu.com/p/137082457)

🔹 [Invariant Risk Minimization](https://arxiv.org/pdf/1907.02893.pdf) Introduction is good! :+1: :fire: :boom:   [slide](https://bayesgroup.github.io/bmml_sem/2019/Kodryan_Invariant%20Risk%20Minimization.pdf) [information theoretic view](https://www.inference.vc/invariant-risk-minimization/)

To learn invariances across environments, find a data representation such that the optimal classifier on top of that representation matches for all environments.

🔹 [Out-of-Distribution Generalization via Risk Extrapolation](https://arxiv.org/pdf/2003.00688.pdf) :+1: :fire:

REx can be viewed as encouraging robustness over affine combinations of training risks, by encouraging strict equality between training risks.

🔹 [OUT-OF-DISTRIBUTION GENERALIZATION ANALYSIS VIA INFLUENCE FUNCTION](https://arxiv.org/pdf/2101.08521.pdf) :fire:

if a learnt model fθˆ manage to simultaneously achieve small Vγˆ|θˆ and high accuracy over E_test, it should have good OOD accuracy.

🔹 [EMPIRICAL OR INVARIANT RISK MINIMIZATION? A SAMPLE COMPLEXITY PERSPECTIVE](https://openreview.net/pdf?id=jrA5GAccy_) :droplet:  ​

🔹 [Invariant Rationalization](http://proceedings.mlr.press/v119/chang20c/chang20c.pdf) :+1: :fire:  :boom:

MMI can be problematic because it picks up spurious correlations between the input features and the output. Instead, we introduce a game-theoretic invariant rationalization criterion where the rationales are constrained to enable the same predictor to be optimal across different environments.  

🔹 [Invariant Risk Minimization Games](http://proceedings.mlr.press/v119/ahuja20a/ahuja20a.pdf) :droplet:  ​

 ​ ​

:o: influence function

- 🔹 [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730.pdf) :fire: :droplet:
  
  Upweighting a training point; Perturbing a training input; Efficiently calculating influence :droplet: ;  
  
  - 🔹 [INFLUENCE FUNCTIONS IN DEEP LEARNING ARE FRAGILE](https://arxiv.org/pdf/2006.14651.pdf) :+1: :fire: :+1:  
  
    non-convexity of the loss function ---  different initializations; parameters might be very large --- substantial Taylor’s approximation error of the loss function; computationally very expensive ---  approximate inverse-Hessian Vector product techniques which might be erroneous;  different architectures can have different loss landscape geometries near the optimal model parameters, leading to varying influence estimates.
  
  - 🔹 [On the Accuracy of Influence Functions for Measuring Group Effects](https://arxiv.org/pdf/1905.13289.pdf) :+1:  ​
  
    when measuring the change in test prediction or test loss, influence is additive.
  
  :o: do-calculate ---> causual inference (Interventions) ---> counterfactuals
  
  see [inFERENCe's blog](https://www.inference.vc/causal-inference-3-counterfactuals/) :+1: :fire: :boom:   the intervention conditional p(y|do(X=x^))p(y|do(X=x^)) is the average of counterfactuals over the obserevable population.  

🔹 [Soft-Robust Actor-Critic Policy-Gradient](https://arxiv.org/pdf/1803.04848.pdf) :confused:  ​

Robust RL has shown that by considering the worst case scenario, robust policies can be overly conservative. Soft-Robust Actor Critic (SR-AC) learns an optimal policy with respect to a distribution over an uncertainty set and stays robust to model uncertainty but avoids the conservativeness of robust strategies.

🔹 [A Game-Theoretic Perspective of Generalization in Reinforcement Learning](https://arxiv.org/pdf/2208.03650.pdf) :fire: :fire:

We propose a game-theoretic framework for the generalization in reinforcement learning, named GiRL, where an RL agent is trained against an adversary over a set of tasks, where the adversary can manipulate the distributions over tasks within a given threshold.

<a name="anchor-irl"></a>

## IL (IRL)

- [Inverse RL & Apprenticeship Learning](https://thegradient.pub/learning-from-humans-what-is-inverse-reinforcement-learning/#:~:text=Inverse%20reinforcement%20learning%20(IRL)%2C,the%20task%20of%20autonomous%20driving.), PPT-levine([1](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_12_irl.pdf):+1: [2](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/inverseRL.pdf)), Medium([1](https://towardsdatascience.com/inverse-reinforcement-learning-6453b7cdc90d) [2](https://medium.com/@jonathan_hui/rl-inverse-reinforcement-learning-56c739acfb5a)),

  🔹 [Apprenticeship Learning via Inverse Reinforcement Learning](http://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Abbeel+Ng:2004.pdf) :+1:   [Maximum Entropy Inverse Reinforcement Learning](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)   [Maximum Entropy Deep Inverse Reinforcement Learning](https://arxiv.org/pdf/1507.04888.pdf)

  🔹  [Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/pdf/1603.00448.pdf) :fire: :+1:  

  🔹 [A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models](https://arxiv.org/pdf/1611.03852.pdf) :volcano: :sweat_drops: :sweat_drops:   [zhihu](https://zhuanlan.zhihu.com/p/72691529) :+1:  ​

  🔹 [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf) :fire: :volcano: :boom: :sweat_drops:  [zhihu](https://zhuanlan.zhihu.com/p/60327435) :droplet:  ​

  IRL is a dual of an occupancy measure matching problem; The induced optimal policy is the primal optimum.

  🔹 [Visual Adversarial Imitation Learning using Variational Models](https://proceedings.neurips.cc/paper/2021/file/1796a48fa1968edd5c5d10d42c7b1813-Paper.pdf) :fire: 🌋 

  V-MAIL:  learns a model of the environment, which serves as a strong self-supervision signal for visual representation learning and mitigates distribution shift by enabling synthetic on-policy rollouts using the model.

  🔹 [Latent Policies for Adversarial Imitation Learning](https://arxiv.org/pdf/2206.11299.pdf) :fire: 

  LAPAL: We use an action encoder-decoder model to obtain a low-dimensional latent action space and train a LAtent Policy using Adversarial imitation Learning (LAPAL).

  🔹 [LEARNING ROBUST REWARDS WITH ADVERSARIAL INVERSE REINFORCEMENT LEARNING](https://arxiv.org/pdf/1710.11248.pdf) :fire::volcano: :boom: :sweat_drops:

  AIRL: Part of the challenge is that IRL is an ill-defined problem, since there are many optimal policies that can explain a set of demonstrations, and many rewards that can explain an optimal policy. The maximum entropy (MaxEnt) IRL framework introduced by Ziebart et al. (2008) handles the former ambiguity, but the latter ambiguity means that IRL algorithms have difficulty distinguishing **the true reward functions from those shaped by the environment dynamics** (THE REWARD AMBIGUITY PROBLEM).  -- **DISENTANGLING REWARDS FROM DYNAMICS.**

  🔹 [Adversarially Robust Imitation Learning](https://proceedings.mlr.press/v164/wang22d/wang22d.pdf) 🌋 

  ARIL: physical attack; sensory attack. 

  🔹 [Robust Imitation of Diverse Behaviors](https://proceedings.neurips.cc/paper/2017/file/044a23cadb567653eb51d4eb40acaa88-Paper.pdf) :fire: 

   VAE+GAN: a new version of GAIL that (1) is much more robust than the purely-supervised controller, especially with few demonstrations, and (2) avoids mode collapse, capturing many diverse behaviors when GAIL on its own does not.

  🔹 [OFF-POLICY ADVERSARIAL INVERSE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2005.01138.pdf)

  🔹 [A Primer on Maximum Causal Entropy Inverse Reinforcement Learning](https://arxiv.org/pdf/2203.11409.pdf) :droplet:  💦 

  🔹 [ADVERSARIAL IMITATION VIA VARIATIONAL INVERSE REINFORCEMENT LEARNING](https://arxiv.org/pdf/1809.06404.pdf) :+1: :fire: :droplet:

  Our method simultaneously learns empowerment through variational information maximization along with the reward and policy under the adversarial learning formulation.  

  🔹 [A Divergence Minimization Perspective on Imitation Learning Methods](http://proceedings.mlr.press/v100/ghasemipour20a/ghasemipour20a.pdf) :sweat_drops: :+1: :volcano:  ​ ​

  State-Marginal Matching. we present a unified probabilistic perspective on IL algorithms based on divergence minimization.

  🔹 [f-IRL: Inverse Reinforcement Learning via State Marginal Matching](https://arxiv.org/pdf/2011.04709.pdf) :sweat_drops:

  🔹 [Imitation Learning as f-Divergence Minimization](https://arxiv.org/pdf/1905.12888.pdf) :sweat_drops:  ​

  🔹 [Offline Imitation Learning with a Misspecified Simulator](https://proceedings.neurips.cc/paper/2020/file/60cb558c40e4f18479664069d9642d5a-Paper.pdf) :+1:  ​

  learn pi in the condition of a few expert demonstrations and a simulator with misspecified dynamics.

  🔹 [Inverse Constrained Reinforcement Learning](https://arxiv.org/pdf/2011.09999.pdf) :fire:  :boom:  

  The main task (“do this”) is often quite easy to encode in the form of a simple nominal reward function. In this work, we focus on learning the constraint part (“do not do that”) from provided expert demonstrations and using it in conjunction with the nominal reward function to train RL agents.

  🔹 [PRIMAL WASSERSTEIN IMITATION LEARNING](https://arxiv.org/pdf/2006.04678.pdf) :fire: :volcano:

  We present Imitation Learning as a distribution matching problem and introduce a reward function which is based on an upper bound of the Wasserstein distance between the state-action distributions of the agent and the expert.  

  🔹 [Robust Inverse Reinforcement Learning under Transition Dynamics Mismatch](https://arxiv.org/pdf/2007.01174.pdf) :fire:

  We consider the Maximum Causal Entropy (MCE) IRL learner model and provide a tight upper bound on the learner’s performance degradation based on the `1-distance between the transition dynamics of the expert and the learner.

  🔹 [XIRL: Cross-embodiment Inverse Reinforcement Learning](https://proceedings.mlr.press/v164/zakka22a/zakka22a.pdf) 

  leverages temporal cycleconsistency constraints to learn deep visual embeddings that capture task progression from offline videos of demonstrations across multiple expert agents, each performing the same task differently due to embodiment differences. 

  🔹 [Deterministic and Discriminative Imitation (D2-Imitation): Revisiting Adversarial Imitation for Sample Efficiency](https://arxiv.org/pdf/2112.06054.pdf) :fire: :volcano:

  Deterministic and Discriminative Imitation (D2-Imitation) operates by first partitioning samples into two replay buffers and then learning a deterministic policy via off-policy reinforcement learning.

  🔹 [Learning Sparse Rewarded Tasks from Sub-Optimal Demonstrations](https://arxiv.org/pdf/2004.00530.pdf) :fire:

  We propose Self-Adaptive Imitation Learning (SAIL) that can achieve (near) optimal performance given only a limited number of sub-optimal demonstrations for highly challenging sparse reward tasks.  reward = log pi_1/pi_2;

  🔹 [Model-Based Imitation Learning Using Entropy Regularization of Model and Policy](https://arxiv.org/pdf/2206.10101.pdf) 👍 🔥 🌋

  MB-ERIL: A policy discriminator distinguishes the actions generated by a robot from expert ones, and a model discriminator distinguishes the counterfactual state transitions generated by the model from the actual ones.

  🔹 [Robust Imitation Learning against Variations in Environment Dynamics](https://arxiv.org/pdf/2206.09314.pdf) 👍 🔥 🌋

  RIME: Our framework effectively deals with environments with varying dynamics by imitating multiple experts in sampled environment dynamics to enhance the robustness in general variations in environment dynamics

  🔹 [Learning Multi-Task Transferable Rewards via Variational Inverse Reinforcement Learning](https://arxiv.org/pdf/2206.09498.pdf) :fire: :fire: 

  Our proposed method derives the variational lower bound of the situational mutual information to optimize it. We simultaneously learn the transferable multi-task reward function and policy by adding an induced term to the objective function.
  
   ​

  🔹 [UNDERSTANDING THE RELATION BETWEEN MAXIMUM-ENTROPY INVERSE REINFORCEMENT LEARNING AND BEHAVIOUR CLONING](https://openreview.net/pdf?id=rkeXrIIt_4)

  🔹 [Disagreement-Regularized Imitation Learning](https://openreview.net/forum?id=rkgbYyHtwB)

  🔹 [Intrinsic Reward Driven Imitation Learning via Generative Model](https://arxiv.org/pdf/2006.15061.pdf) :+1: :fire:

  Combines a backward action encoding and a forward dynamics model into one generative solution. Moreover, our model generates a family of intrinsic rewards, enabling the imitation agent to do samplingbased self-supervised exploration in the environment.  Outperform the expert.  

  🔹 [REGULARIZED INVERSE REINFORCEMENT LEARNING](https://openreview.net/pdf?id=HgLO8yalfwc) :sweat_drops:

  🔹 [Variational Inverse Control with Events: A General Framework for Data-Driven Reward Definition](https://proceedings.neurips.cc/paper/2018/file/c9319967c038f9b923068dabdf60cfe3-Paper.pdf) :volcano: variational inverse control with events (**VICE**), which generalizes inverse reinforcement learning methods  :volcano: :droplet:  ​ ​

  🔹 [Meta-Inverse Reinforcement Learning with Probabilistic Context Variables](https://arxiv.org/pdf/1909.09314.pdf) :droplet:

  we propose a deep latent variable model that is capable of learning rewards from demonstrations of distinct but related tasks in an unsupervised way. Critically, our model can infer rewards for new, structurally-similar tasks from a single demonstration.  

  🔹 [Domain Adaptive Imitation Learning](https://arxiv.org/pdf/1910.00105.pdf) :fire: :+1: :volcano:

  In the alignment step we execute a novel unsupervised MDP alignment algorithm, GAMA, to learn state and action correspondences from unpaired, unaligned demonstrations. In the adaptation step we leverage the correspondences to zero-shot imitate tasks across domains.

  🔹 [ADAIL: Adaptive Adversarial Imitation Learning](https://arxiv.org/pdf/2008.12647.pdf) :+1: :fire: :volcano:

  the discriminator may either simply **use the embodiment or dynamics to infer whether it is evaluating expert behavior**, and as a consequence fails to provide a meaningful reward signal. we condition our policy on a learned dynamics embedding and we employ a domain-adversarial loss to learn a dynamics-invariant discriminator.

  🔹 [Generative Adversarial Imitation from Observation](https://arxiv.org/pdf/1807.06158.pdf) :sweat_drops: :+1: :fire: :volcano:  ​ ​ ​

  From a high-level perspective, in imitation from observation, the goal is to enable the agent to extract what the task is by observing some state sequences.  GAIfO

  🔹 [MobILE: Model-Based Imitation Learning From Observation Alone](https://openreview.net/pdf?id=_Rtm4rYnIIL) :+1: :fire: :volcano:

  Imitation Learning from Observation Alone (ILFO).  MobILE involves carefully trading off strategic exploration against imitation - this is achieved by integrating the idea of optimism in the face of uncertainty into the distribution matching imitation learning (IL) framework.

  🔹 [IMITATION LEARNING FROM OBSERVATIONS UNDER TRANSITION MODEL DISPARITY](https://openreview.net/pdf?id=twv2QlJhXzo) :fire: :+1:

  We consider ILO where the expert and the learner agents operate in different environments (dynamics). We propose an AILO that trains an intermediary policy in the learner environment and uses it as a surrogate expert for the learner.

  🔹 [CROSS-DOMAIN IMITATION LEARNING VIA OPTIMAL TRANSPORT](https://openreview.net/pdf?id=xP3cPq2hQC) :fire:

  We propose Gromov-Wasserstein Imitation Learning (GWIL), a method for cross-domain imitation that uses the Gromov Wasserstein distance to align and compare states between the different spaces of the agents.

  🔹 [An Imitation from Observation Approach to Transfer Learning with Dynamics Mismatch](https://papers.nips.cc/paper/2020/file/28f248e9279ac845995c4e9f8af35c2b-Paper.pdf) :+1: :fire: :volcano:  ​

  learning the grounded action transformation can be seen as an IfO problem; GARAT: learn an action transformation policy for transfer learning with dynamics mismatch.     we focus on the paradigm of simulator grounding, which modifies the source environment’s dynamics to more closely match the target environment dynamics using a relatively small amount of target environment data.

  🔹 [HYAR: ADDRESSING DISCRETE-CONTINUOUS ACTION REINFORCEMENT LEARNING VIA HYBRID ACTION EPRESENTATION](https://openreview.net/pdf?id=64trBbOhdGU)
  
  We propose Hybrid Action Representation (HyAR) to learn a compact and decodable latent representation space for the original hybrid action space.
  
  🔹 [STATE ALIGNMENT-BASED IMITATION LEARNING](https://openreview.net/pdf?id=rylrdxHFDr) :+1: :fire:  ​
  
  Consider an imitation learning problem that the imitator and the expert have different dynamics models. The state alignment comes from both local and global perspectives and we combine them into a reinforcement learning framework by a regularized policy update objective. ifo
  
  🔹 [Strictly Batch Imitation Learning by Energy-based Distribution Matching](https://proceedings.neurips.cc//paper/2020/file/524f141e189d2a00968c3d48cadd4159-Paper.pdf) :fire: :boom: :sweat_drops:  ​
  
   ​By identifying parameterizations of the (discriminative) model of a policy with the (generative) energy function for state distributions, EDM yields a simple but effective solution that equivalently minimizes a divergence between the occupancy measure for the demonstrator and a model thereof for the imitator.
  
  🔹 [SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards](https://arxiv.org/pdf/1905.11108.pdf) :+1: :fire: :volcano:  ​
  
  SQIL is equivalent to a variant of behavioral cloning (BC) that uses regularization to overcome state distribution shift. We accomplish this by giving the agent a constant reward of r = +1 for matching the demonstrated action in a demonstrated state, and a constant reward of r = 0 for all other behavior.
  
  🔹 [IQ-Learn: Inverse soft-Q Learning for Imitation](https://arxiv.org/pdf/2106.12142.pdf) :+1: :fire: :volcano: :droplet:
  
  We introduce a method for dynamics-aware IL which avoids adversarial training by learning a single Q-function, implicitly representing both reward and policy.  
  
  🔹 [Boosted and Reward-regularized Classification for Apprenticeship Learning](https://www.cristal.univ-lille.fr/~pietquin/pdf/AAMAS_2014_BPMGOP.pdf) :fire:  ​ ​
  
  MultiClass Classification and the Large Margin Approach.
  
  🔹 [IMITATION LEARNING VIA OFF-POLICY DISTRIBUTION MATCHING](https://arxiv.org/pdf/1912.05032.pdf) :+1: :fire: :boom:  :volcano:  ​
  
  These prior distribution matching approaches possess two limitations (On-policy; Separate RL optimization).  ---> OFF-POLICY FORMULATION OF THE KL-DIVERGENCE. ---> VALUEDICE: IMITATION LEARNING WITH IMPLICIT REWARDS. (OPE)

  🔹 [TRANSFERABLE REWARD LEARNING BY DYNAMICS-AGNOSTIC DISCRIMINATOR ENSEMBLE](https://arxiv.org/pdf/2206.00238.pdf) :fire:

  DARL: learns a dynamics-agnostic discriminator on a latent space mapped from the original state-action space. To reduce the reliance of the discriminator on policies, the reward function is represented as an ensemble of the discriminators during training.
  
  🔹 [Imitation Learning from Observations by Minimizing Inverse Dynamics Disagreement](https://papers.nips.cc/paper/2019/file/ed3d2c21991e3bef5e069713af9fa6ca-Paper.pdf) :+1: :fire: :boom:  ​
  
  the gap between LfD and LfO actually lies in the disagreement of inverse dynamics models between the imitator and the expert, if following the modeling approach of GAIL.  ifo  IDDM
  
  🔹 [Off-Policy Imitation Learning from Observations](https://arxiv.org/pdf/2102.13185.pdf) :volcano: :sweat_drops: :fire:  :boom:  ​
  
  OPOLO (Off POlicy Learning from Observations)!  ifo // lfo  // ope // mode-covering (Forward Distribution Matching) // mode-seeking // dice // LfD // LfO  ​
  
  🔹 [Imitation Learning by State-Only Distribution Matching](https://arxiv.org/pdf/2202.04332.pdf) :fire: :fire:
  
  LfO: We propose a non-adversarial learning-from-observations approach, together with an interpretable convergence and performance metric.

  🔹 [Plan Your Target and Learn Your Skills: Transferable State-Only Imitation Learning via Decoupled Policy Optimization](https://arxiv.org/pdf/2203.02214.pdf) :fire: 🌋 

  We propose Decoupled Policy Optimization (DePO) for transferable state-only imitation learning, which decouples the state-to-action mapping policy into a state-to-state mapping state planner and a state-pair-to-action mapping inverse dynamics model. [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/3b8a614226a953a8cd9526fca6fe9ba5.png) 
  
  🔹 [IMITATION LEARNING BY REINFORCEMENT LEARNING](https://openreview.net/pdf?id=1zwleytEpYx) :fire: :fire:
  
  We show that, for deterministic experts, imitation learning can be done by reduction to reinforcement learning with a stationary reward.
  
  🔹 [Imitation by Predicting Observations](https://arxiv.org/pdf/2107.03851.pdf) :fire: :volcano:
  
  LfO: FORM (“Future Observation Reward Model”) is derived from an inverse RL objective and imitates using a model of expert behavior learned by generative modelling of the expert’s observations, without needing ground truth actions.  
  
  🔹 [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://arxiv.org/pdf/2104.02180.pdf) :no_mouth:
  
  we presented an adversarial learning system for physics based character animation that enables characters to imitate diverse behaviors from large unstructured datasets, without the need for motion planners or other mechanisms for clip selection.  

  🔹 [ARC - Actor Residual Critic for Adversarial Imitation Learning](https://arxiv.org/pdf/2206.02095.pdf) 😶

  We leverage the differentiability property of the AIL reward function and formulate a class of Actor Residual Critic (ARC) RL algorithms that draw a parallel to the standard AC algorithms in RL and uses a residual critic, C function to approximate only the discounted future return (excluding the immediate reward). 

  🔹 [AUTO-ENCODING INVERSE REINFORCEMENT LEARNING](https://openreview.net/pdf?id=OCgCYv7KGZe) 🌋 

  AEIRL: utilizes the reconstruction error of an auto-encoder as the learning signal, which provides more information for optimizing policies, compared to the binary logistic loss.

  🔹 [Auto-Encoding Adversarial Imitation Learning](https://arxiv.org/pdf/2206.11004.pdf) 

  AEAIL: 

  
🔹 [Reinforced Imitation Learning by Free Energy Principle](https://arxiv.org/pdf/2107.11811.pdf) :droplet:

🔹 [Error Bounds of Imitating Policies and Environments](https://arxiv.org/pdf/2010.11876.pdf)  :volcano:  :sweat_drops:  

🔹 [What Matters for Adversarial Imitation Learning?](https://openreview.net/pdf?id=-OrwaD3bG91)

🔹 [Distributionally Robust Imitation Learning](https://openreview.net/pdf?id=4JiZIwTnXty) :+1:  :fire: :droplet:

This paper studies Distributionally Robust Imitation Learning (DROIL) and establishes a close connection between DROIL and Maximum Entropy Inverse Reinforcement Learning.

🔹 [Provable Representation Learning for Imitation Learning via Bi-level Optimization](https://arxiv.org/pdf/2002.10544.pdf)

🔹 [Provable Representation Learning for Imitation with Contrastive Fourier Features](https://arxiv.org/pdf/2105.12272.pdf) :fire: :boom:

We derive a representation learning objective that provides an upper bound on the performance difference between the target policy and a lowdimensional policy trained with max-likelihood, and this bound is tight regardless of whether the target policy itself exhibits low-dimensional structure.

🔹 [TRAIL: NEAR-OPTIMAL IMITATION LEARNING WITH SUBOPTIMAL DATA](https://arxiv.org/pdf/2110.14770.pdf) :fire: :volcano:  

TRAIL (Transition-Reparametrized Actions for Imitation Learning): We present training objectives that use offline datasets to learn a factored transition model whose structure enables the extraction of a latent action space. Our theoretical analysis shows that the learned latent action space can boost the sample-efficiency of downstream imitation learning, effectively reducing the need for large near-optimal expert datasets through the use of auxiliary non-expert data.

🔹 [Imitation Learning via Differentiable Physics](https://arxiv.org/pdf/2206.04873.pdf) :fire:

ILD: incorporates the differentiable physics simulator as a physics prior into its computational graph for policy learning.

🔹 [Of Moments and Matching: A Game-Theoretic Framework for Closing the Imitation Gap](https://arxiv.org/pdf/2103.03236.pdf) 🌋 

AdVIL, AdRIL, and DAeQuIL: 

🔹 [Generalizable Imitation Learning from Observation via Inferring Goal Proximity](https://proceedings.neurips.cc/paper/2021/file/868b7df964b1af24c8c0a9e43a330c6a-Paper.pdf) :+1: :fire: 

we learn a goal proximity function (task proress) and utilize it as a dense reward for policy learning. 

🔹 [Show me the Way: Intrinsic Motivation from Demonstrations](https://arxiv.org/pdf/2006.12917.pdf) 😶 

extracting an intrinsic bonus from the demonstrations.

- Adding Noise

  🔹 [Learning from Suboptimal Demonstration via Self-Supervised Reward Regression](https://arxiv.org/pdf/2010.11723.pdf) :+1: :fire:  

  Recent attempts to learn from sub-optimal demonstration leverage pairwise rankings and following the Luce-Shepard rule. However, we show these approaches make incorrect assumptions and thus suffer from brittle, degraded performance. We overcome these limitations in developing a novel approach that **bootstraps off suboptimal demonstrations to synthesize optimality-parameterized data** to train an idealized reward function.  

  🔹 [Robust Imitation Learning from Noisy Demonstrations](https://arxiv.org/pdf/2010.10181.pdf) :fire:  :volcano:

  In this paper, we first theoretically show that robust imitation learning can be achieved by optimizing a classification risk with a symmetric loss. Based on this theoretical finding, we then propose a new imitation learning method that optimizes the classification risk by effectively combining pseudo-labeling with co-training.

  🔹 [Imitation Learning from Imperfect Demonstration](http://proceedings.mlr.press/v97/wu19a/wu19a.pdf) :fire: :volcano: :+1:  ​

  a novel approach that utilizes confidence scores, which describe the quality of demonstrations. two-step importance weighting imitation learning (2IWIL) and generative adversarial imitation learning with imperfect demonstration and confidence (IC-GAIL), based on the idea of reweighting.

  🔹 [Variational Imitation Learning with Diverse-quality Demonstrations](http://proceedings.mlr.press/v119/tangkaratt20a/tangkaratt20a.pdf) :fire: :droplet:

  VILD: We show that simple quality-estimation approaches might fail due to compounding error, and fix this issue by jointly estimating both the quality and reward using a variational approach.
  
  🔹 [BEHAVIORAL CLONING FROM NOISY DEMONSTRATIONS](https://openreview.net/pdf?id=zrT3HcsWSAt) :volcano: :sweat_drops:
  
  we propose an imitation learning algorithm to address the problem without any environment interactions and annotations associated with the non-optimal demonstrations.

  🔹 [Robust Imitation Learning from Corrupted Demonstrations](https://arxiv.org/pdf/2201.12594.pdf) :fire: 🌋

  We propose a novel robust algorithm by minimizing a Median-of-Means (MOM) objective which guarantees the accurate estimation of policy, even in the presence of constant fraction of outliers.

  🔹 [Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality](https://proceedings.neurips.cc/paper/2021/file/670e8a43b246801ca1eaca97b3e19189-Paper.pdf) :+1: :fire: 🌋 

  CAIL: learns a well-performing policy from confidence-reweighted demonstrations, while using an outer loss to track the performance of our model and to learn the confidence.

  🔹 [Imitation Learning by Estimating Expertise of Demonstrators](https://arxiv.org/pdf/2202.01288.pdf) :fire: :volcano:

  ILEED: We develop and optimize a joint model over a learned policy and expertise levels of the demonstrators. This enables our model to learn from the optimal behavior and filter out the suboptimal behavior of each demonstrator.

  🔹 [Learning to Weight Imperfect Demonstrations](http://proceedings.mlr.press/v139/wang21aa/wang21aa.pdf) 🌋 

  We provide a rigorous mathematical analysis, presenting that the weights of demonstrations can be exactly determined by combining the discriminator and agent policy in GAIL.

  🔹 [Robust Adversarial Imitation Learning via Adaptively-Selected Demonstrations](https://web.archive.org/web/20210812214931id_/https://www.ijcai.org/proceedings/2021/0434.pdf) :fire: 

  SAIL: good demonstrations can be adaptively selected for training while bad demonstrations are abandoned. 


  
  🔹 [Policy Learning Using Weak Supervision](https://arxiv.org/pdf/2010.01748.pdf) :volcano: :fire:
  
  PeerRL: We treat the “weak supervision” as imperfect information coming from a peer agent, and evaluate the learning agent’s policy based on a “correlated agreement” with the peer agent’s policy (instead of simple agreements).  

 🔹 [Rethinking Importance Weighting for Transfer Learning](https://arxiv.org/pdf/2112.10157.pdf) :volcano:

We review recent advances based on joint and dynamic importance predictor estimation. Furthermore, we introduce a method of causal mechanism transfer that incorporates causal structure in TL. 

🔹 [Inverse Decision Modeling: Learning Interpretable Representations of Behavior](http://proceedings.mlr.press/v139/jarrett21a/jarrett21a.pdf) :fire: 🌋 

We develop an expressive, unifying perspective on inverse decision modeling: a framework for learning parameterized representations of sequential decision behavior. 

🔹 [DISCRIMINATOR-ACTOR-CRITIC: ADDRESSING SAMPLE INEFFICIENCY AND REWARD BIAS IN ADVERSARIAL IMITATION LEARNING](https://arxiv.org/pdf/1809.02925.pdf) :no_mouth: 

DAC: To address reward bias, we propose a simple mechanism whereby the rewards for absorbing states are also learned; To improve sample efficiency, we perform off-policy training.

🔹 [Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations](https://arxiv.org/pdf/1904.06387.pdf) :+1: 

T-REX: a reward learning technique for high-dimensional tasks that can learn to extrapolate intent from suboptimal ranked demonstrations.

🔹 [Better-than-Demonstrator Imitation Learning via Automatically-Ranked Demonstrations](https://arxiv.org/pdf/1907.03976.pdf)

D-REX: a ranking-based reward learning algorithm that does not require ranked demonstrations, which injects noise into a policy learned through behavioral cloning to automatically generate ranked demonstrations.

🔹 [Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences](https://arxiv.org/pdf/2002.09089.pdf) 

🔹 [A Ranking Game for Imitation Learning](https://arxiv.org/pdf/2202.03481.pdf) :fire: :fire: 

The rankinggame additionally affords a broader perspective of imitation, going beyond using only expert demonstrations, and utilizing rankings/preferences over suboptimal behaviors.

🔹 [Learning Multimodal Rewards from Rankings](https://proceedings.mlr.press/v164/myers22a/myers22a.pdf) :fire: 

We formulate the multimodal reward learning as a mixture learning problem and develop a novel ranking-based learning approach, where the experts are only required to rank a given set of trajectories.



🔹 [Semi-Supervised Imitation Learning of Team Policies from Suboptimal Demonstrations](https://arxiv.org/pdf/2205.02959.pdf) 

BTIL: 

🔹 [Learning Reward Functions from Scale Feedback](https://arxiv.org/pdf/2110.00284.pdf) :+1: 

 Instead of a strict question on which of the two proposed trajectories the user prefers, we allow for more nuanced feedback using a slider bar.

🔹 [Interactive Learning from Policy-Dependent Human Feedback](http://proceedings.mlr.press/v70/macglashan17a/macglashan17a.pdf) 

COACH: 

🔹 [Towards Sample-efficient Apprenticeship Learning from Suboptimal Demonstration](https://arxiv.org/pdf/2110.04347.pdf) 😶 

SSRR, S3RR: noise-performance curve fitting --> regresses a reward function of trajectory states and actions. 

🔹 [BASIS FOR INTENTIONS: EFFICIENT INVERSE REINFORCEMENT LEARNING USING PAST EXPERIENCE](https://arxiv.org/pdf/2208.04919.pdf) :fire: 🌋 

BASIS, which leverages multi-task RL pre-training and successor features to allow an agent to build a strong basis for intentions that spans the space of possible goals in a given domain.

🔹 [POSITIVE-UNLABELED REWARD LEARNING](https://arxiv.org/pdf/1911.00459.pdf) :fire: 🌋 

PURL: we connect these two classes of reward learning methods (GAIL, SL) to positiveunlabeled (PU) learning, and we show that by applying a large-scale PU learning algorithm to the reward learning problem, we can address both the reward underand over-estimation problems simultaneously.

🔹 [Combating False Negatives in Adversarial Imitation Learning](https://arxiv.org/pdf/2002.00412.pdf) :+1: 

Fake Conditioning

🔹 [Task-Relevant Adversarial Imitation Learning](https://arxiv.org/pdf/1910.01077.pdf) :fire: :fire: 

TRAIL proposes to constrain the GAIL discriminator such that it is not able to distinguish between certain, preselected expert and agent observations which do not contain task behavior.

- Multiple-Intent
  
  🔹 [LiMIIRL: Lightweight Multiple-Intent Inverse Reinforcement Learning](https://arxiv.org/pdf/2106.01777.pdf) :fire: 

  Multiple-Intent Inverse Reinforcement Learning (MI-IRL) seeks to find a reward function ensemble to rationalize demonstrations of different but unlabelled intents. Within the popular expectation maximization (EM) framework for learning probabilistic MI-IRL models, we present a warm-start strategy based on up-front clustering of the demonstrations in feature space.

- Meta IRL

  🔹 [Meta-Inverse Reinforcement Learning with Probabilistic Context Variables](https://arxiv.org/pdf/1909.09314.pdf) :fire: 

  PEMIRL: we propose a deep latent variable model that is capable of learning rewards from demonstrations of distinct but related tasks in an unsupervised way.

- LfL 
  
  🔹 [Inverse Reinforcement Learning from a Gradient-based Learner](https://proceedings.neurips.cc/paper/2020/file/19aa6c6fb4ba9fcf39e893ff1fd5b5bd-Paper.pdf) 👍 :fire: 

  LOGEL: the goal is to recover the reward function being optimized by an agent, given a sequence of policies produced during learning.

- RL From Preferences

  🔹 [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/pdf/1706.03741.pdf) :fire:

  We explore goals defined in terms of (non-expert) human preferences between pairs of trajectory segments.  

  🔹 [Reward learning from human preferences and demonstrations in Atari](https://arxiv.org/pdf/1811.06521.pdf) :no_mouth:

  We combine two approaches to learning from human feedback: expert demonstrations and trajectory preferences.

  🔹 [PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training](https://arxiv.org/pdf/2106.05091.pdf)

  We present an off-policy, interactive RL algorithm that capitalizes on the strengths of both feedback and off-policy learning.

  🔹 [Skill Preferences: Learning to Extract and Execute Robotic Skills from Human Feedback](https://arxiv.org/pdf/2108.05382.pdf) :no_mouth:

  We introduce Skill Preferences (SkiP), an algorithm that incorporates human feedback to extract skills from (noisy) offline data and utilize those skills to solve downstream tasks.

  🔹 [Widening the Pipeline in Human-Guided Reinforcement Learning with Explanation and Context-Aware Data Augmentation](https://papers.nips.cc/paper/2021/file/b6f8dc086b2d60c5856e4ff517060392-Paper.pdf)

  We focus on the task of learning from feedback, in which the human trainer not only gives binary evaluative "good" or "bad" feedback for queried state-action pairs, but also provides a visual explanation by annotating relevant features in images. We then propose EXPAND (EXPlanation AugmeNted feeDback) to encourage the model to encode task-relevant features.

  🔹 [Offline Preference-Based Apprenticeship Learning](https://arxiv.org/pdf/2107.09251.pdf) :no_mouth:

  OPAL: Given a database consisting of trajectories without reward labels, we query an expert for preference labels over trajectory segments from the database, learn a reward function from preferences, and then perform offline RL using rewards provided by the learned reward function.

  🔹 [Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data](https://arxiv.org/pdf/2111.06849.pdf) :+1: :fire: :volcano:

  This paper introduces a novel strategy called Adaptive Pseudo Augmentation (APA) to encourage healthy competition between the generator and the discriminator. APA alleviates overfitting by employing the generator itself to augment the real data distribution with generated images, which deceives the discriminator adaptively.

  🔹 [B-Pref: Benchmarking Preference-Based Reinforcement Learning](https://arxiv.org/pdf/2111.03026.pdf) :volcano:
  
  We introduce B-Pref: a benchmark specially designed for preference-based RL.
  
  🔹 [Batch Reinforcement Learning from Crowds](https://arxiv.org/pdf/2111.04279.pdf) :no_mouth:
  
   This paper tackles a critical challenge that emerged when collecting data from non-expert humans: the noise in preferences.
  
  🔹 [Dueling RL: Reinforcement Learning with Trajectory Preferences](https://arxiv.org/pdf/2111.04850.pdf)
  
  🔹 [SURF: SEMI-SUPERVISED REWARD LEARNING WITH DATA AUGMENTATION FOR FEEDBACK-EFFICIENT PREFERENCE-BASED REINFORCEMENT LEARNING](https://openreview.net/pdf?id=TfhfZLQ2EJO) :no_mouth:
  
  We present SURF, a semi-supervised reward learning framework that utilizes a large amount of unlabeled samples with data augmentation, where we infer pseudo-labels of the unlabeled samples based on the confidence of the preference predictor.
  
  🔹 [Teachable Reinforcement Learning via Advice Distillation](https://arxiv.org/pdf/2203.11197.pdf) :no_mouth:
  
  We propose a new supervision paradigm for interactive learning based on “teachable” decision-making systems that learn from structured advice provided by an external teacher.  
  
  🔹 [ReIL: A Framework for Reinforced Intervention-based Imitation Learning](https://arxiv.org/pdf/2203.15390.pdf) :+1:
  
  We introduce Reinforced Interventionbased Learning (ReIL), a framework consisting of a general intervention-based learning algorithm and a multi-task imitation learning model aimed at enabling non-expert users to train agents in real environments with little supervision or fine tuning.  

  🔹 [Learning to summarize from human feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf) 😶 

  🔹 [MORAL: Aligning AI with Human Norms through Multi-Objective Reinforced Active Learning](https://arxiv.org/pdf/2201.00012.pdf) 😶 

   Through maintaining a distribution over scalarization weights, our approach is able to interactively tune a deep RL agent towards a variety of preferences, while eliminating the need for computing multiple policies. 

- Reward Comparison; PBRS (potential-based reward shaping) 

  🔹 [QUANTIFYING DIFFERENCES IN REWARD FUNCTIONS](https://arxiv.org/pdf/2006.13900.pdf) :fire: 🌋 

  We introduce the Equivalent-Policy Invariant Comparison (EPIC) distance to quantify the difference between two reward functions directly, without a policy optimization step.
  
  🔹 [DYNAMICS-AWARE COMPARISON OF LEARNED REWARD FUNCTIONS](https://arxiv.org/pdf/2201.10081.pdf) :+1: 🌋 

  DARD uses an approximate transition model of the environment to transform reward functions into a form that allows for comparisons that are invariant to reward shaping while only evaluating reward functions on transitions close to their training distribution. 

  🔹 [Preprocessing Reward Functions for Interpretability](https://arxiv.org/pdf/2203.13553.pdf) :+1: 🔥 

  We propose exploiting the intrinsic structure of reward functions by first preprocessing them into simpler but equivalent reward functions, which are then visualized. 

  🔹 [Understanding Learned Reward Functions](https://arxiv.org/pdf/2012.05862.pdf) 😶 

  We have explored the use of saliency maps and counterfactuals to understand learned reward functions.

  🔹 [Explicable Reward Design for Reinforcement Learning Agents](https://machineteaching.mpi-sws.org/files/papers/neurips21_explicable-reward-design.pdf) 🌋 💧 

  EXPRD allows us to appropriately balance informativeness and sparseness while guaranteeing that an optimal policy induced by the function belongs to a set of target policies. EXPRD builds upon an informativeness criterion that captures the (sub-)optimality of target policies at different time horizons from any given starting state.

  🔹 [Automatic shaping and decomposition of reward functions](https://dspace.mit.edu/bitstream/handle/1721.1/35890/MIT-CSAIL-TR-2007-010.pdf?sequence=1&isAllowed=y) 

  🔹 [Dynamic Potential-Based Reward Shaping](https://eprints.whiterose.ac.uk/75121/2/p433_devlin.pdf) :+1:

  We have proven that a dynamic potential function can be used to shape an agent without altering its optimal policy. 

  🔹 [Expressing Arbitrary Reward Functions as Potential-Based Advice](https://ai.vub.ac.be/sites/default/files/aaai-anna-draft_3.pdf) :fire: 

  DPBA: Potential-based reward shaping is a way to provide the agent with a specific form of additional reward, with the guarantee of policy invariance. In this work we give a novel way to incorporate an arbitrary reward function with the same guarantee, by implicitly translating it into the specific form of dynamic advice potentials, which are maintained as an auxiliary value function learnt at the same time.

  🔹 [Useful Policy Invariant Shaping from Arbitrary Advice](https://arxiv.org/pdf/2011.01297.pdf) :fire: :+1: 

  PIES biases the agent’s policy toward the advice at the start of the learning, when the agent is the most in need of guidance. Over time, PIES gradually decays this bias to zero, ensuring policy invariance.

  🔹 [Policy Transfer using Reward Shaping](https://ai.vub.ac.be/~tbrys/publications/Brys2015AAMAS.pdf) 👍 
  
  We presented a novel approach to policy transfer, encoding the transferred policy as a dynamic potential-based reward shaping function, benefiting from all the theory behind reward shaping.

  🔹 [Reward prediction for representation learning and reward shaping](https://arxiv.org/pdf/2105.03172.pdf) :+1: 

  Using our representation for preprocessing high-dimensional observations, as well as using the predictor for reward shaping.

- Inverse constrain learning (ICL)
  
  🔹 [Learning Soft Constraints From Constrained Expert Demonstrations](https://arxiv.org/pdf/2206.01311.pdf) :fire: 

   We consider the setting where the reward function is given, and the constraints are unknown, and propose a method that is able to recoverthese constraints satisfactorily from the expert data.

- Delayed reward 
  
  🔹 



  
<a name="anchor-offline "></a>

## Offline RL

  🔹 [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/pdf/2005.01643.pdf)​  :boom: :boom:  ​ :droplet:  ​

  Offline RL with dynamic programming: distributional shift; policy constraints; uncertainty estimation; conservative Q-learning and Pessimistic Value-function;

  <https://danieltakeshi.github.io/2020/06/28/offline-rl/> :sweat_drops:

  <https://ai.googleblog.com/2020/08/tackling-open-challenges-in-offline.html> :sweat_drops:

  <https://sites.google.com/view/offlinerltutorial-neurips2020/home> :sweat_drops:

🔹 [D4RL: DATASETS FOR DEEP DATA-DRIVEN REINFORCEMENT LEARNING](https://arxiv.org/pdf/2004.07219.pdf) :volcano: :volcano:

examples of such properties include: datasets generated via hand-designed controllers and human demonstrators, multitask datasets where an agent performs different tasks in the same environment, and datasets collected with mixtures of policies.  

🔹 [d3rlpy: An Offline Deep Reinforcement Learning Library](https://arxiv.org/pdf/2111.03788.pdf) :volcano:

🔹 [A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://arxiv.org/pdf/2203.01387.pdf) :droplet:

🔹 [An Optimistic Perspective on Offline Reinforcement Learning](http://proceedings.mlr.press/v119/agarwal20c/agarwal20c.pdf) :+1:  ​

To enhance generalization in the offline setting, we present Random Ensemble Mixture (REM), a robust Q-learning algorithm that enforces optimal Bellman consistency on random convex combinations of multiple Q-value estimates.

  🔹 [OPAL: OFFLINE PRIMITIVE DISCOVERY FOR ACCELERATING OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2010.13611.pdf) :boom:  when presented with offline data composed of a variety of behaviors, an effective way to leverage this data is to extract a continuous space of recurring and temporally extended primitive behaviors before using these primitives for downstream task learning. OFFLINE unsupervised RL.

  🔹 [Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization](https://arxiv.org/pdf/2006.03647.pdf) :+1: :volcano:  :boom:

🔹 [TOWARDS DEPLOYMENT-EFFICIENT REINFORCEMENT LEARNING: LOWER BOUND AND OPTIMALITY](https://openreview.net/pdf?id=ccWaPGl9Hq) :confused:

We propose such a formulation for deployment-efficient RL (DE-RL) from an “optimization with constraints” perspective: we are interested in exploring an MDP and obtaining a near-optimal policy within minimal deployment complexity, whereas in each deployment the policy can sample a large batch of data.


🔹 [MUSBO: Model-based Uncertainty Regularized and Sample Efficient Batch Optimization for Deployment Constrained Reinforcement Learning](https://arxiv.org/pdf/2102.11448.pdf) :fire: :+1: 

Our framework discovers novel and high quality samples for each deployment to enable efficient data collection. During each offline training session, we bootstrap the policy update by quantifying the amount of uncertainty within our collected data.

  🔹 [BENCHMARKS FOR DEEP OFF-POLICY EVALUATION](https://openreview.net/pdf?id=kWSeGEeHvF8) :+1:

  🔹 [KEEP DOING WHAT WORKED: BEHAVIOR MODELLING PRIORS FOR OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2002.08396.pdf) :+1: :fire: :boom:

  It admits the use of data generated by arbitrary behavior policies and uses a learned prior – the advantage-weighted behavior model (ABM) – to bias the RL policy towards actions that have previously been executed and are likely to be successful on the new task.

  extrapolation or bootstrapping errors:  (Fujimoto et al., 2018; Kumar et al., 2019)

  🔹 [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf) :fire: :boom: :volcano:  ​ ​ ​

  BCQ: We introduce a novel class of off-policy algorithms, batch-constrained reinforcement learning, which restricts the action space in order to force the agent towards behaving close to on-policy with respect to a subset of the given data.

  🔹 [Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://arxiv.org/pdf/1906.00949.pdf) :fire: :boom: :droplet:  ​

BEAR: We identify bootstrapping error as a key source of instability in current methods. Bootstrapping error is due to bootstrapping from actions that lie outside of the training data distribution, and it accumulates via the Bellman backup operator.  

  🔹 [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf) :+1: :fire: :volcano: :sweat_drops: :boom:

  conservative Q-learning (CQL), which aims to address these limitations by learning a conservative Q-function such that the expected value of a policy under this Q-function lower-bounds its true value.

  🔹 [Mildly Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.04745.pdf) :+1: :fire: 🌋 

  We propose Mildly Conservative Q-learning (MCQ), where OOD actions are actively trained by assigning them proper pseudo Q values. 

🔹 [Constraints Penalized Q-Learning for Safe Offline Reinforcement Learning](https://arxiv.org/pdf/2107.09003.pdf) :+1: :fire: :volcano:

We show that naïve approaches that combine techniques from safe RL and offline RL can only learn sub-optimal solutions. We thus develop a simple yet effective algorithm, Constraints Penalized Q-Learning (CPQ), to solve the problem.

🔹 [Conservative Offline Distributional Reinforcement Learning](https://arxiv.org/pdf/2107.06106.pdf) :sweat_drops:

CODAC:

🔹 [OFFLINE REINFORCEMENT LEARNING HANDS-ON](https://arxiv.org/pdf/2011.14379.pdf)

🔹 [Supervised Off-Policy Ranking](https://arxiv.org/pdf/2107.01360.pdf) :+1: 

SOPR: aims to rank a set of target policies based on supervised learning by leveraging off-policy data and policies with known performance. [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/eb46c61f91aab8c2b002b288485fc118_Cjgm5dl.png) 

🔹 [Conservative Data Sharing for Multi-Task Offline Reinforcement Learning](https://arxiv.org/pdf/2109.08128.pdf) :fire: :volcano: :boom:  

Conservative data sharing (CDS): We develop a simple technique for data-sharing in multi-task offline RL that routes data based on the improvement over the task-specific data.

🔹 [Data Sharing without Rewards in Multi-Task Offline Reinforcement Learning](https://openreview.net/pdf?id=c7SmcWAd74W)

Conservative unsupervised data sharing (CUDS): under a binary-reward assumption, simply utilizing data from other tasks with constant reward labels can not only provide substantial improvement over only using the single-task data and previously proposed success classifiers, but it can also reach comparable performance to baselines that take advantage of the oracle multi-task reward information.

🔹 [Switch Trajectory Transformer with Distributional Value Approximation for Multi-Task Reinforcement Learning](https://arxiv.org/pdf/2203.07413.pdf)

🔹 [How to Leverage Unlabeled Data in Offline Reinforcement Learning](https://arxiv.org/pdf/2202.01741.pdf) :fire: :volcano:

We provide extensive theoretical and empirical analysis that illustrates how it trades off reward bias, sample complexity and distributional shift, often leading to good results. We characterize conditions under which this simple strategy is effective, and further show that extending it with a simple reweighting approach can further alleviate the bias introduced by using incorrect reward labels.

🔹 [Is Pessimism Provably Efficient for Offline RL?](http://proceedings.mlr.press/v139/jin21e/jin21e.pdf) :fire: :volcano: :fire:

Pessimistic value iteration algorithm (PEVI): incorporates a penalty function (pessimism) into the value iteration algorithm. The penalty function simply flips the sign of the bonus function (optimism) for promoting exploration in online RL.  We decompose the suboptimality of any policy into three sources: the spurious correlation, intrinsic uncertainty, and optimization error.

🔹 [PESSIMISTIC MODEL-BASED OFFLINE REINFORCEMENT LEARNING UNDER PARTIAL COVERAGE](https://openreview.net/pdf?id=tyrJsbKAe6) :volcano:

Constrained Pessimistic Policy Optimization (CPPO): We study model-based offline RL with function approximation under partial coverage. We show that for the model-based setting, realizability in function class and partial coverage together are enough to learn a policy that is comparable to any policies covered by the offline distribution.

🔹 [Corruption-Robust Offline Reinforcement Learning](https://arxiv.org/pdf/2106.06630.pdf) :confused:

🔹 [Bellman-consistent Pessimism for Offline Reinforcement Learning](https://arxiv.org/pdf/2106.06926.pdf) :fire: :confused:

We introduce the notion of Bellman-consistent pessimism for general function approximation: instead of calculating a point-wise lower bound for the value function, we implement pessimism at the initial state over the set of functions consistent with the Bellman equations.

🔹 [Provably Good Batch Reinforcement Learning Without Great Exploration](https://arxiv.org/pdf/2007.08202.pdf) :fire: :fire:

We show that a small modification to Bellman optimality and evaluation back-up to take a more conservative update can have much stronger guarantees. In certain settings, they can find the approximately best policy within the state-action space explored by the batch data, without requiring a priori assumptions of concentrability.

🔹 [Pessimistic Q-Learning for Offline Reinforcement Learning: Towards Optimal Sample Complexity](https://arxiv.org/pdf/2202.13890.pdf) :fire: :droplet:

LCB-Q (value iteration with lower confidence bounds): We study a pessimistic variant of Q-learning in the context of finite-horizon Markov decision processes, and characterize its sample complexity under the single policy concentrability assumption which does not require the full coverage of the state-action space.

🔹 [Policy Finetuning: Bridging Sample-Efficient Offline and Online Reinforcement Learning](https://arxiv.org/pdf/2106.04895.pdf) :fire: :droplet:

This paper initiates the theoretical study of policy finetuning, that is, online RL where the learner has additional access to a “reference policy” \miu close to the optimal policy \pi⋆ in a certain sense.

🔹 [Towards Instance-Optimal Offline Reinforcement Learning with Pessimism](https://papers.nips.cc/paper/2021/file/212ab20dbdf4191cbcdcf015511783f4-Paper.pdf)

🔹 [Provable Benefits of Actor-Critic Methods for Offline Reinforcement Learning](https://papers.nips.cc/paper/2021/file/713fd63d76c8a57b16fc433fb4ae718a-Paper.pdf)

Pessimistic Actor Critic for Learning without Exploration (PACLE)

🔹 [WHEN SHOULD OFFLINE REINFORCEMENT LEARNING BE PREFERRED OVER BEHAVIORAL CLONING?](https://openreview.net/pdf?id=AP1MKT37rJ) :volcano: :volcano:

under what environment and dataset conditions can an offline RL method outperform BC with an equal amount of expert data, even when BC is a natural choice?  [Should I Run Offline Reinforcement Learning or Behavioral Cloning?]

🔹 [PESSIMISTIC BOOTSTRAPPING FOR UNCERTAINTY-DRIVEN OFFLINE REINFORCEMENT LEARNING](https://openreview.net/pdf?id=Y4cs1Z3HnqL) :+1: :fire:

PBRL: We propose Pessimistic Bootstrapping for offline RL (PBRL), a purely uncertainty-driven offline algorithm without explicit policy constraints. Specifically, PBRL conducts uncertainty quantification via the disagreement of bootstrapped Q-functions, and performs pessimistic updates by penalizing the value function based on the estimated uncertainty.

🔹 [UNCERTAINTY REGULARIZED POLICY LEARNING FOR OFFLINE REINFORCEMENT LEARNING](https://openreview.net/pdf?id=rwSWaS_tGgG) :+1:

 Uncertainty Regularized Policy Learning (URPL): URPL adds an uncertainty regularization term in the policy learning objective to enforce to learn a more stable policy under the offline setting. Moreover, we further use the uncertainty regularization term as a surrogate metric indicating the potential performance of a policy.

🔹 [Model-Based Offline Meta-Reinforcement Learning with Regularization](https://openreview.net/pdf?id=EBn0uInJZWh) :+1: :fire: :volcano:

We explore model-based offline Meta-RL with regularized Policy Optimization (MerPO), which learns a meta-model for efficient task structure inference and an informative meta-policy for safe exploration of out-of-distribution state-actions.

🔹 [BATCH REINFORCEMENT LEARNING THROUGH CONTINUATION METHOD](https://openreview.net/pdf?id=po-DLlBuAuz) :fire:

We propose a simple yet effective approach, soft policy iteration algorithm through continuation method to alleviate two challenges in policy optimization under batch reinforcement learning: (1) highly non-smooth objective function which is difficult to optimize (2) high variance in value estimates.

🔹 [SCORE: SPURIOUS CORRELATION REDUCTION FOR OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2110.12468.pdf) :+1: :fire: :fire:

We propose a practical and theoretically guaranteed algorithm SCORE that reduces spurious correlations by combing an uncertainty penalty into policy evaluation. We show that this is consistent with the pessimism principle studied in theory, and the proposed algorithm converges to the optimal policy with a sublinear rate under mild assumptions.

🔹 [Why so pessimistic? Estimating uncertainties for offline rl through ensembles, and why their independence matters](https://offline-rl-neurips.github.io/2021/pdf/51.pdf) :+1:

our proposed MSG algorithm advocates for using independently learned ensembles, without sharing of target values, and this import design decision is supported by empirical evidence.

  🔹 [S4RL: Surprisingly Simple Self-Supervision for Offline Reinforcement Learning](https://arxiv.org/pdf/2103.06326.pdf) :no_mouth:

   utilizes data augmentations from states to learn value functions that are better at generalizing and extrapolating when deployed in the environment.

  🔹 [Actionable Models: Unsupervised Offline Reinforcement Learning of Robotic Skills](https://arxiv.org/pdf/2104.07749.pdf) :+1: :fire: :boom:

  learning a functional understanding of the environment by learning to reach any goal state in a given dataset. We employ goal-conditioned Qlearning with hindsight relabeling and develop several techniques that enable training in a particularly challenging offline setting.

  🔹 [Behavior Regularized Offline Reinforcement Learning](https://arxiv.org/pdf/1911.11361.pdf) :fire:  :boom: :+1:

  we introduce a general framework, behavior regularized actor critic (BRAC), to empirically evaluate recently proposed methods as well as a number of simple baselines across a variety of offline continuous control tasks.

🔹 [BRAC+: Improved Behavior Regularized Actor Critic for Offline Reinforcement Learning](https://arxiv.org/pdf/2110.00894.pdf) :fire:

We improved the behavior regularized offline RL by proposing a low-variance upper bound of the KL divergence estimator to reduce variance and gradient penalized policy evaluation such that the learned Q functions are guaranteed to converge.

  🔹 [Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble](https://arxiv.org/pdf/2107.00591.pdf) :no_mouth:  

  we observe that state-action distribution shift may lead to severe bootstrap error during fine-tuning, which destroys the good initial policy obtained via offline RL.

  🔹 [Experience Replay with Likelihood-free Importance Weights](https://proceedings.mlr.press/v168/sinha22a/sinha22a.pdf) :fire: 🌋 

  To balance bias (from off-policy experiences) and variance (from on-policy experiences), we use a likelihood-free density ratio estimator between onpolicy and off-policy experiences, and use the learned ratios as the prioritization weights.

  🔹 [MOORe: Model-based Offline-to-Online Reinforcement Learning](https://arxiv.org/pdf/2201.10070.pdf) :fire: 

  employs a prioritized sampling scheme that can dynamically adjust the offline and online data for smooth and efficient online adaptation of the policy. 

  🔹 [Offline Meta-Reinforcement Learning with Online Self-Supervision](https://arxiv.org/pdf/2107.03974.pdf) :+1: :fire:

  Unlike the online setting, the adaptation and exploration strategies cannot effectively adapt to each other, resulting in poor performance. we propose a hybrid offline meta-RL algorithm, which uses offline data with rewards to meta-train an adaptive policy, and then collects additional unsupervised online data, without any ground truth reward labels, to bridge this distribution shift problem.

  🔹 [Offline Meta-Reinforcement Learning with Advantage Weighting](http://proceedings.mlr.press/v139/mitchell21a/mitchell21a.pdf) :fire:  ​

Targeting the offline meta-RL setting, we propose Meta-Actor Critic with Advantage Weighting (MACAW), an optimization-based meta-learning algorithm that uses simple, supervised regression objectives for both the inner and outer loop of meta-training.

  🔹 [Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning](https://arxiv.org/pdf/2206.10442.pdf) :fire: :fire: 

  CORRO: which decreases the influence of behavior policies on task representations while supporting tasks that differ in reward function and transition dynamics.

  🔹 [AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/pdf/2006.09359.pdf) :+1: :fire: :volcano:

  we systematically analyze why this problem (offline + online) is so challenging, and propose an algorithm that combines sample efficient dynamic programming with maximum likelihood policy updates, providing a simple and effective framework that is able to leverage large amounts of offline data and then quickly perform online fine-tuning of RL policies.

  🔹 [Critic Regularized Regression](https://arxiv.org/pdf/2006.15134.pdf) :+1: :fire: :volcano:  ​ ​

  CRR: Our alg. can be seen as a form of filtered behavioral cloning where data is selected based on information contained in the policy’s Q-fun. we do not rely on observed returns for adv. estimation.

  🔹 [Exponentially Weighted Imitation Learning for Batched Historical Data](https://ai.tencent.com/ailab/media/publications/exponentially-weighted-imitation.pdf) :+1: :fire: :volcano:  ​ ​

  MARWIL: we propose a monotonic advantage reweighted imitation learning strategy that is applicable to problems with complex nonlinear function approximation and works well with hybrid (discrete and continuous) action space.

  🔹 [BAIL: Best-Action Imitation Learning for Batch Deep Reinforcement Learning](https://arxiv.org/pdf/1910.12179.pdf) :+1:  :volcano:  ​

 BAIL learns a V function, uses the V function to select actions it believes to be high-performing, and then uses those actions to train a policy network using imitation learning.

  🔹 [Offline RL Without Off-Policy Evaluation](https://arxiv.org/pdf/2106.08909.pdf) :droplet:  :fire:

a unified algorithmic template for offline RL algorithms as offline approximate modified policy iteration.

🔹 [MODEL-BASED OFFLINE PLANNING](https://arxiv.org/pdf/2008.05556.pdf) :fire:

MBOP:  Learning dynamics, action priors, and values; MBOP-Policy; MBOP-Trajopt.

🔹 [Model-Based Offline Planning with Trajectory Pruning](https://arxiv.org/pdf/2105.07351.pdf) :+1: :fire:

MOPP: MOPP avoids over-restrictive planning while enabling offline learning by encouraging more aggressive trajectory rollout guided by the learned behavior policy, and prunes out problematic trajectories by evaluating the uncertainty of the dynamics model.

🔹 [Model-based Offline Policy Optimization with Distribution Correcting Regularization](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_581.pdf) :+1: :volcano:

 DROP (density ratio regularized offline policy learning ) estimates the density ratio between model-rollouts distribution and offline data distribution via the DICE framework, and then regularizes the model predicted rewards with the ratio for pessimistic policy learning.

🔹 [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/pdf/2106.06860.pdf) :+1: :fire: :volcano:  ​ ​ ​

 We find that we can match the performance of state-of-the-art offline RL algorithms by simply adding a behavior cloning term to the policy update of an online RL algorithm and normalizing the data.

🔹 [POPO: Pessimistic Offline Policy Optimization](https://arxiv.org/pdf/2012.13682.pdf) :confused:  ​

Distributional value functions.

🔹 [Offline Reinforcement Learning as Anti-Exploration](https://arxiv.org/pdf/2106.06431.pdf) :+1: :fire:  ​

The core idea is to subtract a prediction-based exploration bonus from the reward, instead of adding it for exploration.  

🔹 [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/pdf/2005.13239.pdf) :+1:  :fire: :volcano: :boom:  

we propose to modify the existing model-based RL methods by applying them with rewards artificially penalized by the uncertainty of the dynamics. We theoretically show that the algorithm maximizes a lower bound of the policy’s return under the true MDP. We also characterize the trade-off between the gain and risk of leaving the support of the batch data.

🔹 [MOReL: Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2005.05951.pdf) :volcano: :boom: :droplet:  

This framework consists of two steps: (a) learning a pessimistic MDP (P-MDP) using the offline dataset; (b) learning a near-optimal policy in this P-MDP.

🔹 [COMBO: Conservative Offline Model-Based Policy Optimization](https://arxiv.org/pdf/2102.08363.pdf) :fire: :volcano: :boom:   ​ ​ ​

This results in a conservative estimate of the value function for out-of-support state-action tuples, without requiring explicit uncertainty estimation.

🔹 [Regularizing a Model-based Policy Stationary Distribution to Stabilize Offline Reinforcement Learning](https://arxiv.org/pdf/2206.07166.pdf) 🌋 :fire: 

SDM-GAN: we regularize the undiscounted stationary distribution of the current policy towards the offline data during the policy optimization process. [[ppt]](https://icml.cc/media/icml-2022/Slides/18409.pdf) 

🔹 [HYBRID VALUE ESTIMATION FOR OFF-POLICY EVALUATION AND OFFLINE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2206.02000.pdf) 🌋 

We propose Hybrid Value Estimation (HVE) to perform a more accurate value function estimation in the offline setting. It automatically adjusts the step length parameter to get a bias-variance trade-off.

🔹 [DROMO: Distributionally Robust Offline Model-based Policy Optimization](https://arxiv.org/pdf/2109.07275.pdf) :fire:

To extend the basic idea of regularization without uncertainty quantification, we propose distributionally robust offline model-based policy optimization (DROMO), which leverages the ideas in distributionally robust optimization to penalize a broader range of out-of-distribution state-action pairs beyond the standard empirical out-of-distribution Q-value minimization.

🔹 [Behavioral Priors and Dynamics Models: Improving Performance and Domain Transfer in Offline RL](https://arxiv.org/pdf/2106.09119.pdf)  :volcano:

MABE: By adaptive behavioral prior, we mean a policy that approximates the behavior in the offline dataset while giving more importance to trajectories with high rewards.  

🔹 [Offline Reinforcement Learning with Fisher Divergence Critic Regularization](http://proceedings.mlr.press/v139/kostrikov21a/kostrikov21a.pdf) :fire:  :volcano: :droplet:

We propose using a gradient penalty regularizer for the offset term and demonstrate its equivalence to Fisher divergence regularization, suggesting connections to the score matching and generative energy-based model literature.

🔹 [Uncertainty Weighted Actor-Critic for Offline Reinforcement Learning](https://arxiv.org/pdf/2105.08140.pdf) :+1: :fire:  ​ ​

UWAC: an algorithm that detects OOD state-action pairs and down-weights their contribution in the training objectives accordingly.

🔹 [Model-based Offline Policy Optimization with Distribution Correcting Regularization](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_581.pdf)

🔹 [EMaQ: Expected-Max Q-Learning Operator for Simple Yet Effective Offline and Online RL](http://proceedings.mlr.press/v139/ghasemipour21a/ghasemipour21a.pdf) :fire: :volcano: :droplet:

By introducing the Expect-Max Q-Learning operator, we present a novel theoretical setup that takes into account the proposal distribution µ(a|s) and the number of action samples N, and hence more closely matches the resulting practical algorithm.

🔹 [Lyapunov Density Models: Constraining Distribution Shift in Learning-Based Control](https://arxiv.org/pdf/2206.10524.pdf) :fire: 🌋 

We presented Lyapunov density models (LDMs), a tool that can ensure that an agent remains within the distribution of the training data.

🔹 [OFFLINE REINFORCEMENT LEARNING WITH IN-SAMPLE Q-LEARNING](https://openreview.net/pdf?id=68n2s9ZJWF8) :fire: :fire:  

We presented implicit Q-Learning (IQL), a general algorithm for offline RL that completely avoids any queries to values of out-of-sample actions during training while still enabling multi-step dynamic programming.  Adopting Expectile regression. [old](https://arxiv.org/pdf/2110.06169.pdf)

🔹 [Continuous Doubly Constrained Batch Reinforcement Learning](https://arxiv.org/pdf/2102.09225.pdf) :fire: :fire:

CDC: The first regularizer combats the extra-overestimation bias in regions that are out-of-distribution. The second regularizer is designed to hedge against the adverse effects of policy updates that severly diverge from behavior policy.

🔹 [Believe What You See: Implicit Constraint Approach for Offline Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2106.03400.pdf) :+1: :droplet:

ICQ: we propose a novel offline RL algorithm, named Implicit Constraint Q-learning (ICQ), which effectively alleviates the extrapolation error by only trusting the state-action pairs given in the dataset for value estimation.

🔹 [Offline Model-based Adaptable Policy Learning](https://openreview.net/pdf?id=lrdXc17jm6) :+1: :fire: :volcano:

MAPLE tries to model all possible transition dynamics in the out-of-support regions. A context encoder RNN is trained to produce latent codes given the episode history, and the encoder and policy are jointly optimized to maximize average performance across a large ensemble of pretained dynamics models.

🔹 [Supported Policy Optimization for Offline Reinforcement Learning](https://arxiv.org/pdf/2202.06239.pdf) :no_mouth:

We present Supported Policy OpTimization (SPOT), which is directly derived from the theoretical formalization of the density based support constraint. SPOT adopts a VAEbased density estimator to explicitly model the support set of behavior policy.

🔹 [Weighted model estimation for offline model-based reinforcement learning](https://proceedings.neurips.cc/paper/2021/file/949694a5059302e7283073b502f094d7-Paper.pdf) :fire:

This paper considers weighting with the state-action distribution ratio of offline data and simulated future data, which can be estimated relatively easily by standard density ratio estimation techniques for supervised learning.

🔹 [Batch Reinforcement Learning with Hyperparameter Gradients](http://ailab.kaist.ac.kr/papers/pdfs/LLVKK2020.pdf) :+1: :fire: :volcano: :fire:

BOPAH: Unlike prior work where this trade-off is controlled by hand-tuned hyperparameters (in a generalized KL-regularized RL framework), we propose a novel batch reinforcement learning approach, batch optimization of policy and hyperparameter (BOPAH), that uses a gradient-based optimization of the hyperparameter using held-out data.

🔹 [OFFLINE REINFORCEMENT LEARNING WITH VALU-EBASED EPISODIC MEMORY](https://arxiv.org/pdf/2110.09796.pdf) :volcano: :droplet: :fire: 

We present a new offline V -learning method, EVL (expectile V -learning), and a novel offline RL framework, VEM (Value-based Episodic Memory). EVL learns the value function through the trade-offs between imitation learning and optimal value learning. VEM uses a memory-based planning scheme to enhance advantage estimation and conduct policy learning in a regression manner.

🔹 [Offline Reinforcement Learning with Soft Behavior Regularization](https://arxiv.org/pdf/2110.07395.pdf) :fire: :volcano:

Soft Behavior-regularized Actor Critic (SBAC): we design a new behavior regularization scheme for offline RL that enables policy improvement guarantee and state-dependent policy regularization.

🔹 [Offline Reinforcement Learning with Pseudometric Learning](https://arxiv.org/pdf/2103.01948.pdf) :+1: :fire: :volcano:

 In the presence of function approximation, and under the assumption of limited coverage of the state-action space of the environment, it is necessary to enforce the policy to visit state-action pairs close to the support of logged transitions. In this work, we propose an iterative procedure to learn a pseudometric (closely related to bisimulation metrics) from logged transitions, and use it to define this notion of closeness.  

🔹 [Offline Reinforcement Learning with Reverse Model-based Imagination](https://arxiv.org/pdf/2110.00188.pdf) :fire: :fire:

Reverse Offline Model-based Imagination (ROMI): We learn a reverse dynamics model in conjunction with a novel reverse policy, which can generate rollouts leading to the target goal states within the offline dataset.

🔹 [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/pdf/2110.01548.pdf) :+1: :fire:

we propose an uncertainty-based model-free offline RL method that effectively quantifies the uncertainty of the Q-value estimates by an ensemble of Q-function networks and does not require any estimation or sampling of the data distribution.

🔹 [ROBUST OFFLINE REINFORCEMENT LEARNING FROM LOW-QUALITY DATA](https://openreview.net/pdf?id=uOjm_xqKEoX) :no_mouth:  

AdaPT: we propose an Adaptive Policy constrainT (AdaPT) method, which allows effective exploration on out-ofdistribution actions by imposing an adaptive constraint on the learned policy.

🔹 [Regularized Behavior Value Estimation](https://arxiv.org/pdf/2103.09575.pdf)   :+1:  :fire:  :fire:

R-BVE uses a ranking regularisation term that favours actions in the dataset that lead to successful outcomes. CRR \  MPO.

🔹 [Active Offline Policy Selection](https://arxiv.org/pdf/2106.10251.pdf) :+1: :volcano:  ​ ​

Gaussian process over policy values; Kernel; Active offline policy selection with Bayesian optimization.  We proposed a BO solution that integrates OPE estimates with evaluations obtained by interacting with env.

🔹 [Offline Policy Selection under Uncertainty](https://arxiv.org/pdf/2012.06919.pdf) :sweat_drops:  ​

🔹 [Offline Learning from Demonstrations and Unlabeled Experience](https://arxiv.org/pdf/2011.13885.pdf) :no_mouth: :+1: :fire:

We proposed offline reinforced imitation learning (ORIL) to enable learning from both demonstrations and a large unlabeled set of experiences without reward annotations.  

🔹 [Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations](https://proceedings.mlr.press/v162/xu22l/xu22l.pdf) :fire: :volcano: :boom:

DWBC: We introduce an additional discriminator to distinguish expert and non-expert data, we propose a cooperation strategy to boost the performance of both tasks, this will result in a new policy learning objective and surprisingly, we find its equivalence to a generalized BC objective, where the outputs of discriminator serve as the weights of the BC loss function.

🔹 [Discriminator-Guided Model-Based Offline Imitation Learning](https://arxiv.org/pdf/2207.00244.pdf) :fire: 

(DMIL) framework, which introduces a discriminator to simultaneously distinguish the dynamics correctness and suboptimality of model rollout data against real expert demonstrations.

🔹 [Offline Preference-Based Apprenticeship Learning](https://arxiv.org/pdf/2107.09251.pdf) :+1:  ​

OPAL: Given a database consisting of trajectories without reward labels, we query an expert for preference labels over trajectory segments from the database, learn a reward function from preferences, and then perform offline RL using rewards provided by the learned reward function.

🔹 [Semi-supervised reward learning for offline reinforcement learning](https://arxiv.org/pdf/2012.06899.pdf) :no_mouth:  ​

We train a reward function on a pre-recorded dataset, use it to label the data and do offline RL.

🔹 [LEARNING VALUE FUNCTIONS FROM UNDIRECTED STATE-ONLY EXPERIENCE](https://openreview.net/pdf?id=6Pe99Juo9gd) :fire:

This paper tackles the problem of learning value functions from undirected state only experience (state transitions without action labels i.e. (s, s' , r) tuples).

🔹 [Offline Inverse Reinforcement Learning](https://arxiv.org/pdf/2106.05068.pdf)

🔹  [Augmented World Models Facilitate Zero-Shot Dynamics Generalization From a Single Offline Environment](https://arxiv.org/pdf/2104.05632.pdf) :+1: :fire: :volcano:

We augment a learned dynamics model with simple transformations that seek to capture potential changes in physical properties of the robot, leading to more robust policies.

🔹 [SEMI-PARAMETRIC TOPOLOGICAL MEMORY FOR NAVIGATION](https://arxiv.org/pdf/1803.00653.pdf) 

🔹 [Mapping State Space using Landmarks for Universal Goal Reaching](https://arxiv.org/pdf/1908.05451.pdf)

🔹 [Search on the Replay Buffer: Bridging Planning and Reinforcement Learning](https://arxiv.org/pdf/1906.05253.pdf)

🔹 [Hallucinative Topological Memory for Zero-Shot Visual Planning](https://arxiv.org/pdf/2002.12336.pdf)

🔹 [Sparse Graphical Memory for Robust Planning](https://arxiv.org/pdf/2003.06417.pdf) 👍

SGM: aggregates states according to a novel two-way consistency objective, adapting classic state aggregation criteria to goal-conditioned RL: two states are redundant when they are interchangeable both as goals and as starting states.

🔹 [Plan2vec: Unsupervised Representation Learning by Latent Plans](https://arxiv.org/pdf/2005.03648.pdf) 😶

Plan2vec constructs a weighted graph on an image dataset using near-neighbor distances, and then extrapolates this local metric to a global embedding by distilling path-integral over planned path. 

🔹 [World Model as a Graph: Learning Latent Landmarks for Planning](https://arxiv.org/pdf/2011.12491.pdf) :+1:

L3P: We devise a novel algorithm to learn latent landmarks that are scattered (in terms of reachability) across the goal space as the nodes on the graph. 

🔹 [Value Memory Graph: A Graph-Structured World Model for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.04384.pdf) :fire:

VMG: we design a graph-structured world model in offline reinforcement learning by building a directed-graph-based Markov decision process (MDP) with rewards allocated to each directed edge as an abstraction of the original continuous environment.

🔹 [GOAL-CONDITIONED BATCH REINFORCEMENT LEARNING FOR ROTATION INVARIANT LOCOMOTION](https://arxiv.org/pdf/2004.08356.pdf) :no_mouth:

🔹 [Offline Meta-Reinforcement Learning for Industrial Insertion](https://arxiv.org/pdf/2110.04276.pdf) :no_mouth:

We introduced an offline meta-RL algorithm, ODA, that can meta-learn an adaptive policy from offline data, quickly adapt based on a small number of user-provided demonstrations for a new task, and then further adapt through online finetuning.

🔹 [Scaling data-driven robotics with reward sketching and batch reinforcement learning](https://arxiv.org/pdf/1909.12200.pdf)

🔹 [OFFLINE RL WITH RESOURCE CONSTRAINED ONLINE DEPLOYMENT](https://arxiv.org/pdf/2110.03165.pdf) :+1:

Resourceconstrained setting: We highlight the performance gap between policies trained using the full offline dataset and policies trained using limited features.

🔹 [Reinforcement Learning from Imperfect Demonstrations](https://arxiv.org/pdf/1802.05313.pdf) :fire:

 We propose  Normalized Actor-Critic (NAC) that effectively normalizes the Q-function, reducing the Q-values of actions unseen in the demonstration data. NAC learns an initial policy network from demonstrations and refines the policy in the environment.

🔹 [Curriculum Offline Imitating Learning](https://openreview.net/pdf?id=q6Kknb68dQf) :fire:

We propose Curriculum Offline Imitation Learning (COIL), which utilizes an experience picking strategy for imitating from adaptive neighboring policies with a higher return, and improves the current policy along curriculum stages.

🔹 [Dealing with the Unknown: Pessimistic Offline Reinforcement Learning](https://arxiv.org/pdf/2111.05440.pdf) :fire:

PessORL: penalize high values at unseen states in the dataset, and to cancel the penalization at in-distribution states.

🔹 [Adversarially Trained Actor Critic for Offline Reinforcement Learning](https://arxiv.org/pdf/2202.02446.pdf) :fire: :volcano:

We propose Adversarially Trained Actor Critic (ATAC) based on a two-player Stackelberg game framing of offline RL: A policy actor competes against an adversarially trained value critic, who finds data-consistent scenarios where the actor is inferior to the data-collection behavior policy. [robust policy improvement] [[POSTER]](https://icml.cc/media/PosterPDFs/ICML%202022/69dd2eff9b6a421d5ce262b093bdab23.png) 

🔹 [RVS: WHAT IS ESSENTIAL FOR OFFLINE RL VIA SUPERVISED LEARNING?](https://arxiv.org/pdf/2112.10751.pdf) :no_mouth:

Simply maximizing likelihood with a two-layer feedforward MLP is competitive with state-of-the-art results of substantially more complex methods based on TD learning or sequence modeling with Transformers. Carefully choosing model capacity (e.g., via regularization or architecture) and choosing which information to condition on (e.g., goals or rewards) are critical for performance.  [THE ESSENTIAL ELEMENTS OF OFFLINE RL VIA SUPERVISED LEARNING](https://openreview.net/pdf?id=S874XAIpkR-)  

🔹 [Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2206.07568.pdf) :+1: :fire: 🌋 

 instead of adding representation learning parts to an existing RL algorithm, we show (contrastive) representation learning methods can be cast as RL algorithms in their own right. 

🔹 [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/pdf/2206.01079.pdf) :fire:

We find that RCSL (return-conditioned SL) returns the optimal policy under a set of assumptions that are stronger than those needed for the more traditional dynamic programming-based algorithms.

🔹 [Implicit Behavioral Cloning](https://arxiv.org/pdf/2109.00137.pdf) :fire: :volcano: :boom:

In this paper we showed that reformulating supervised imitation learning as a conditional energy-based modeling problem, with inference-time implicit regression, often greatly outperforms traditional explicit policy baselines.

🔹 [Implicit Two-Tower Policies](https://arxiv.org/pdf/2208.01191.pdf) :+1: 

Implicit Two-Tower (ITT) policies, where the actions are chosen based on the attention scores of their learnable latent representations with those of the input states.

🔹 [Latent-Variable Advantage-Weighted Policy Optimization for Offline RL](https://arxiv.org/pdf/2203.08949.pdf) :no_mouth:

LAPO: we study an offline RL setup for learning from heterogeneous datasets where trajectories are collected using policies with different purposes, leading to a multi-modal data distribution.

🔹 [AW-Opt: Learning Robotic Skills with Imitation and Reinforcement at Scale](https://proceedings.mlr.press/v164/lu22a/lu22a.pdf) :+1:

Our aim is to test the scalability of prior IL + RL algorithms and devise a system based on detailed empirical experimentation that combines existing components in the most effective and scalable way.

🔹 [Offline RL Policies Should be Trained to be Adaptive](https://arxiv.org/pdf/2207.02200.pdf) :fire: 🌋 

APE-V: optimal policies for offline RL must be adaptive, depending not just on the current state but rather all the transitions seen so far during evaluation.

🔹 [Distance-Sensitive Offline Reinforcement Learning](https://arxiv.org/pdf/2205.11027.pdf) 👍 :fire: 🌋 

We propose a new method, DOGE (Distance-sensitive Offline RL with better GEneralization). DOGE marries dataset geometry with deep function approximators in offline RL, and enables exploitation in generalizable OOD areas rather than strictly constraining policy within data distribution. 

🔹 [RORL: Robust Offline Reinforcement Learning via Conservative Smoothing](https://arxiv.org/pdf/2206.02829.pdf) :+1: :fire: 

We explicitly introduce regularization on the policy and the value function for states near the dataset and additional conservative value estimation on these OOD states.

🔹 [On the Role of Discount Factor in Offline Reinforcement Learning](https://arxiv.org/pdf/2206.03383.pdf) :fire: :fire:

This paper examines two distinct effects of discount factor in offline RL with theoretical analysis, namely the regularization effect and the pessimism effect.

🔹 [LEARNING PSEUDOMETRIC-BASED ACTION REPRESENTATIONS FOR OFFLINE REINFORCEMENT LEARNING](https://openreview.net/pdf?id=naoQDOYsHnS) :fire: 🌋

BMA: This paper proposes an action representation learning framework for offline RL based on a pseudometric, which measures both the behavioral relation and thedata-distributional relation between actions.

🔹 [PLAS: Latent Action Space for Offline Reinforcement Learning](https://arxiv.org/pdf/2011.07213.pdf) :+1:

We propose to simply learn the Policy in the
Latent Action Space (PLAS) such that this requirement (OOD action) is naturally satisfied.

🔹 [Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations](https://arxiv.org/pdf/2206.04779.pdf) 

🔹 [Back to the Manifold: Recovering from Out-of-Distribution States](https://arxiv.org/pdf/2207.08673.pdf) :fire: 🌋 

We alleviate the distributional shift at the deployment time by introducing a recovery policy that brings the agent back to the training manifold whenever it steps out of the in-distribution states, e.g., due to an external perturbation.





🔹 :small_blue_diamond: 
🔹 :small_blue_diamond: 
🔹 :small_blue_diamond: 


:o: Designs from Data | offline MBO

🔹 Designs from Data: Offline Black-Box Optimization via Conservative Training  [see here](https://bair.berkeley.edu/blog/2021/10/25/coms_mbo/)  

🔹 [OFFLINE MODEL-BASED OPTIMIZATION VIA NORMALIZED MAXIMUM LIKELIHOOD ESTIMATION](https://arxiv.org/pdf/2102.07970.pdf) :volcano: :droplet:

we consider data-driven optimization problems where one must maximize a function given only queries at a fixed set of points. provides a principled approach to handling uncertainty and out-of-distribution inputs.

🔹 [Model Inversion Networks for Model-Based Optimization](https://proceedings.neurips.cc/paper/2020/file/373e4c5d8edfa8b74fd4b6791d0cf6dc-Paper.pdf) :fire:

MINs: This work addresses data-driven optimization problems, where the goal is to find an input that maximizes an unknown score or reward function given access to a dataset of inputs with corresponding scores.  

🔹 [RoMA: Robust Model Adaptation for Offline Model-based Optimization](https://openreview.net/pdf?id=VH0TRmnqUc) :+1:

RoMA consists of two steps: (a) a pre-training strategy to robustly train the proxy model and (b) a novel adaptation procedure of the proxy model to have robust estimates for a specific set of candidate solutions.

🔹 [Conservative Objective Models for Effective Offline Model-Based Optimization](http://proceedings.mlr.press/v139/trabucco21a/trabucco21a.pdf) :fire: :volcano:

COMs: We propose conservative objective models (COMs), a method that learns a model of the objective function which lower bounds the actual value of the ground-truth objective on outof-distribution inputs and uses it for optimization.

🔹 [DATA-DRIVEN OFFLINE OPTIMIZATION FOR ARCHITECTING HARDWARE ACCELERATORS](https://arxiv.org/pdf/2110.11346.pdf) :+1:

PRIME: we develop such a data-driven offline optimization method for designing hardware accelerators. PRIME learns a conservative, robust estimate of the desired cost function, utilizes infeasible points and optimizes the design against this estimate without any additional simulator queries during optimization.

🔹 [Conditioning by adaptive sampling for robust design](https://arxiv.org/pdf/1901.10060.pdf) :volcano:

🔹 [DESIGN-BENCH: BENCHMARKS FOR DATA-DRIVEN OFFLINE MODEL-BASED OPTIMIZATION](https://arxiv.org/pdf/2202.08450.pdf) :fire:

Design-Bench, a benchmark for offline MBO with a unified evaluation protocol and reference implementations of recent methods.

🔹 [User-Interactive Offline Reinforcement Learning](https://arxiv.org/pdf/2205.10629.pdf) :fire: :fire: 

We propose an algorithm that allows the user to tune this hyperparameter (the proximity of the learned policy to the original policy) at runtime, thereby overcoming both of the above mentioned issues simultaneously.

🔹 [Comparing Model-free and Model-based Algorithms for Offline Reinforcement Learning](https://arxiv.org/pdf/2201.05433.pdf) 😶 

We compare model-free, model-based, as well as hybrid offline RL approaches on various industrial benchmark (IB) datasets to test the algorithms in settings closer to real world problems, including complex noise and partially observable states.

🔹 [Autofocused oracles for model-based design](https://proceedings.neurips.cc/paper/2020/file/972cda1e62b72640cb7ac702714a115f-Paper.pdf) :fire: :fire: 

we now reformulate the MBD problem as a non-zero-sum game, which suggests an algorithmic strategy for iteratively updating the oracle within any MBO algorithm


<a name="anchor-exploration"></a>

## Exploration

- [Exploration Strategies in Deep Reinforcement Learning](https://lilianweng.github.io/lil-log/2020/06/07/exploration-strategies-in-deep-reinforcement-learning.html) [[chinese]](https://mp.weixin.qq.com/s/FX-1IlIaFDLaQEVFN813jA) :sweat_drops: :fire: :fire:  :boom:

  🔹 [VIME: Variational Information Maximizing Exploration](https://arxiv.org/pdf/1605.09674.pdf) :+1: :punch: :droplet:  ​ ​BNN

   the agent should take actions that maximize the reduction in uncertainty about the dynamics.

  🔹  [Self-Supervised Exploration via Disagreement](https://arxiv.org/pdf/1906.04161.pdf)  :+1:

  an ensemble of dynamics models and incentivize the agent to explore such that the disagreement of those ensembles is maximized.

  🔹 [DORA THE EXPLORER: DIRECTED OUTREACHING REINFORCEMENT ACTION-SELECTION](https://arxiv.org/pdf/1804.04012.pdf) :fire: :+1: :droplet:

  We propose **E-values**, a generalization of counters that can be used to evaluate the propagating exploratory value over state-action trajectories.  [The Hebrew University of Jerusalem] :+1:

  🔹 [EXPLORATION BY RANDOM NETWORK DISTILLATION](https://arxiv.org/pdf/1810.12894.pdf) :fire: :+1: [medium](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938)  :+1:  ​

  based on random network distillation (**RND**) bonus

  🔹 [Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/pdf/1806.03335.pdf) :fire: :boom: :sweat_drops:  ​ ​ ​

  🔹 [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/pdf/1808.04355.pdf) :+1:  ​

  🔹 [NEVER GIVE UP: LEARNING DIRECTED EXPLORATION STRATEGIES](https://arxiv.org/pdf/2002.06038.pdf)  :punch: :+1:

  episodic memory based intrinsic reward using k-nearest neighbors;   self-supervised inverse dynamics model; Universal Value Function Approximators; different degrees of exploration/exploitation;  distributed RL;

  🔹 [Self-Imitation Learning via TrajectoryConditioned Policy for Hard-Exploration Tasks](https://arxiv.org/pdf/1907.10247.pdf) :sweat_drops:  ​

  🔹 [Planning to Explore via Self-Supervised World Models](https://arxiv.org/pdf/2005.05960.pdf)  :fire: :fire: :+1:  ​ ​ ​Experiment is good!  

  a self supervised reinforcement learning agent that tackles both these challenges through a new approach to self-supervised exploration and fast adaptation to new tasks, which need not be known during exploration.  **unlike prior methods which retrospectively compute the novelty of observations after the agent has already reached them**, our agent acts efficiently by leveraging planning to seek out expected future novelty.  

  🔹 [BYOL-Explore: Exploration by Bootstrapped Prediction](https://arxiv.org/pdf/2206.08332.pdf) :fire: 

  BYOL-Explore learns a world representation, the world dynamics, and an exploration policy alltogether by optimizing a single prediction loss in the latent space with no additional auxiliary objective.

  🔹 [Efficient Exploration via State Marginal Matching](https://arxiv.org/pdf/1906.05274.pdf) :fire: :volcano: :droplet:  :boom:  ​

  our work unifies prior exploration methods as performing approximate distribution matching, and explains how state distribution matching can be performed properly

  🔹 hard exploration

  ​ [Provably efficient RL with Rich Observations via Latent State Decoding](https://arxiv.org/pdf/1901.09018.pdf) :confused:  ​

  Block MDP:

  ​ [Provably Efficient Exploration for RL with Unsupervised Learning](https://arxiv.org/pdf/2003.06898.pdf) :confused:  ​

  🔹 [Learning latent state representation for speeding up exploration](https://arxiv.org/pdf/1905.12621.pdf) :+1:

  Prior experience on separate but related tasks help learn representations of the state which are effective at predicting instantaneous rewards.

  🔹 [Self-Imitation Learning](http://proceedings.mlr.press/v80/oh18b/oh18b.pdf) [reward shaping] :+1: :fire:  

  exploiting past good experiences can indirectly drive deep exploration.  we consider exploiting what the agent has experienced, but has not yet learned. Related work: Exploration; Episodic control; Experience replay; Experience replay for actor-critic; Connection between policy gradient and Q-learning; Learning from imperfect demonstrations.  

  🔹 [Generative Adversarial Self-Imitation Learning](https://arxiv.org/pdf/1812.00950.pdf)  [reward shaping] :fire: ​

  GASIL focuses on reproducing past good trajectories, which can potentially make long-term credit assignment easier when rewards are sparse and delayed.

  🔹 [Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration](http://proceedings.mlr.press/v139/han21a/han21a.pdf) :fire: :droplet:

  To take advantage of the previous sample distribution from the replay buffer for sample-efficient exploration, we propose sample-aware entropy regularization which maximizes the entropy of weighted sum of the policy action distribution and the sample action distribution from the replay buf.

  🔹 [LEARNING SELF-IMITATING DIVERSE POLICIES](https://arxiv.org/pdf/1805.10309.pdf) :+1: :fire: :fire:  ​

  We view each policy as a state-action visitation distribution and formulate policy optimization as a divergence minimization problem. We show that with Jensen-Shannon divergence, this divergence minimization problem can be reduced into a policy-gradient algorithm with shaped rewards learned from experience replays.  One approach to achieve better exploration in challenging cases like above is to simultaneously learn *multiple diverse policies* and enforce them to explore different parts of the high dimensional space.  

  🔹 [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm 2016](https://arxiv.org/pdf/1608.04471.pdf)

  🔹 [ADVERSARIALLY GUIDED ACTOR-CRITIC](https://arxiv.org/pdf/2102.04376.pdf) :+1: :fire: :fire:

  While the adversary mimics the actor by minimizing the KL-divergence between their respective action distributions, the actor, in addition to learning to solve the task, tries to differentiate itself from the adversary predictions.

  🔹 [Diversity-Driven Exploration Strategy for Deep Reinforcement Learning](https://arxiv.org/pdf/1802.04564.pdf) :+1: :fire:  ​ ​

  adding a distance measure regularization to the loss function,

  🔹 [Provably Efficient Maximum Entropy Exploration](https://arxiv.org/pdf/1812.02690.pdf) :confused:  

  🔹 [Reward-Free Exploration for Reinforcement Learning](http://proceedings.mlr.press/v119/jin20d/jin20d.pdf) :+1: :fire: :boom:

  *How can we efficiently explore an environment without using any reward information?*  In the exploration phase, the agent first collects trajectories from an MDP M without a pre-specified reward function. After exploration, it is tasked with computing near-optimal policies under for M for a collection of given reward functions.

  🔹 [Rethinking Exploration for Sample-Efficient Policy Learning](https://arxiv.org/pdf/2101.09458.pdf) :+1:  ​

  BBE: bias with finite samples, slow adaptation to decaying bonuses, and lack of optimism on unseen transitions ---> UFO, produces policies that are Unbiased with finite samples, Fast-adapting as the exploration bonus changes, and Optimistic with respect to new transitions.

  🔹 [Provably Efficient Exploration in Policy Optimization](http://proceedings.mlr.press/v119/cai20d/cai20d.pdf) :confused:
  
   design a provably efficient policy optimization algorithm that incorporates exploration.
  
  🔹 [Dynamic Bottleneck for Robust Self-Supervised Exploration](https://arxiv.org/pdf/2110.10735.pdf) :+1: :fire:
  
  We propose a Dynamic Bottleneck (DB) model, which attains a dynamics-relevant representation based on the information-bottleneck principle. Based on the DB model, we further propose DB-bonus, which encourages the agent to explore state-action pairs with high information gain.
  
  🔹 [Principled Exploration via Optimistic Bootstrapping and Backward Induction](https://arxiv.org/pdf/2105.06022.pdf) :+1: :fire:
  
  We propose a principled exploration method for DRL through Optimistic Bootstrapping and Backward Induction (OB2I). OB2I constructs a generalpurpose UCB-bonus through non-parametric bootstrap in DRL. The UCB-bonus estimates the epistemic uncertainty of state-action pairs for optimistic exploration.
  
  🔹 [A Max-Min Entropy Framework for Reinforcement Learning](https://arxiv.org/pdf/2106.10517.pdf) :volcano:  :droplet:
  
  The proposed max-min entropy framework aims to learn to visit states with low entropy and maximize the entropy of these low-entropy states to promote exploration.  
  
  🔹 [Exploration in Deep Reinforcement Learning: A Comprehensive Survey](https://arxiv.org/pdf/2109.06668.pdf) :sweat_drops:
  
  🔹 [HYPERDQN: A RANDOMIZED EXPLORATION METHOD FOR DEEP REINFORCEMENT LEARNING](https://openreview.net/pdf?id=X0nrKAXu7g-) :droplet: :+1:
  
  We present a practical exploration method to address the limitations of RLSVI and BootDQN.
  
  🔹 [Rapid Exploration for Open-World Navigation with Latent Goal Models](https://arxiv.org/pdf/2104.05859.pdf) :+1:
  
  RECON: We use an information bottleneck to regularize the learned policy, giving us (i) a compact visual representation of goals, (ii) improved generalization capabilities, and (iii) a mechanism for sampling feasible goals for exploration.  
  
  🔹 [Better Exploration with Optimistic Actor-Critic](https://arxiv.org/pdf/1910.12807.pdf) :+1:
  
  OAC: we Optimistic Actor Critic, which approximates a lower and upper confidence bound on the state-action value function. This allows us to apply the principle of optimism in the face of uncertainty to perform directed exploration using the upper bound while still using the lower bound to avoid overestimation.

  🔹 [Guided Exploration in Reinforcement Learning via Monte Carlo Critic Optimization](https://arxiv.org/pdf/2206.12674.pdf) 😶 

  An ensemble of Monte Carlo Critics that provides exploratory direction is presented as a controller.

  🔹 [Tactical Optimism and Pessimism for Deep Reinforcement Learning](https://arxiv.org/pdf/2102.03765.pdf) :fire:
  
  TOP: we propose the use of an adaptive approach in which the degree of optimism or pessimism is adjusted dynamically during training. As a consequence of this approach, the optimal degree of optimism can vary across tasks and over the course of a single training run as the model improves.
  
  🔹 [Off-policy Reinforcement Learning with Optimistic Exploration and Distribution Correction](https://arxiv.org/pdf/2110.12081.pdf) :+1:
  
  We adapt the recently introduced DICE framework to learn a distribution correction ratio for off-policy actor-critic training.
  
  🔹 [A Unified Framework for Conservative Exploration](https://arxiv.org/pdf/2106.11692.pdf)
  
   🔹 [A RISK-SENSITIVE POLICY GRADIENT METHOD](https://openreview.net/pdf?id=9rKTy4oZAQt)
  
  🔹 [Policy Gradient for Coherent Risk Measures](https://papers.nips.cc/paper/2015/file/024d7f84fff11dd7e8d9c510137a2381-Paper.pdf)
  
  🔹 [Wasserstein Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2110.07940.pdf) :+1:
  
   By maximizing Wasserstein distance, the agents equipped with different policies may drive themselves to enter different areas of state space and keep as “far” as possible from each other to earn greater diversity.
  
  
  
<a name="anchor-causual"></a>  

## Causal Inference

- Causal inference  [ see more in <a href="#anchor-ood">OOD</a> & [inFERENCe's blog](https://www.inference.vc/causal-inference-3-counterfactuals/) ]

  🔹

- reasoning

  🔹 [CAUSAL DISCOVERY WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/1906.04477.pdf) :no_mouth:  ​

  🔹 [DEEP REINFORCEMENT LEARNING WITH CAUSALITYBASED INTRINSIC REWARD](https://openreview.net/pdf?id=30I4Azqc_oP) :+1:

  The proposed algorithm learns a graph to encode the environmental structure by calculating Average Causal Effect (ACE) between different categories of entities, and an intrinsic reward is given to encourage the agent to interact more with entities belonging to top-ranked categories, which significantly boosts policy learning.
  
  🔹 [Causal Confusion in Imitation Learning](https://arxiv.org/pdf/1905.11979.pdf) :+1: :droplet:
  
  propose a solution to combat it through targeted interventions—either environment interaction or expert queries—to determine the correct causal model.

- cv
  
  🔹 [OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization](https://arxiv.org/pdf/2106.03721.pdf) :fire: 🌋 

  Evaluate OoD generalization algorithms comprehensively on two types of datasets, one dominated by diversity shift and the other dominated by correlation shift.
 ​

<a name="anchor-supervised"></a>  <a name="anchor-goalcon"></a>  

## Supervised RL & Goal-conditioned Policy

  🔹 [LEARNING TO REACH GOALS VIA ITERATED SUPERVISED LEARNING](https://openreview.net/pdf?id=rALA0Xo6yNJ) :no_mouth: :+1:  ​

  GCSL: an agent continually relabels and imitates the trajectories it generates to progressively learn goal-reaching behaviors from scratch.  see more in RVS:  see more in <https://www.youtube.com/watch?v=sVPm7zOrBxM&ab_channel=RAIL>
🔹 [RETHINKING GOAL-CONDITIONED SUPERVISED LEARNING AND ITS CONNECTION TO OFFLINE RL](https://openreview.net/pdf?id=KJztlfGPdwW) :fire: :+1: :volcano:

We propose Weighted GCSL (WGCSL), in which we introduce an advanced compound weight consisting of three parts (1) discounted weight for goal relabeling, (2) goal-conditioned exponential advantage weight, and (3) best advantage weight.  

🔹 [Learning Latent Plans from Play](https://arxiv.org/pdf/1903.01973.pdf) :fire:

Play-GCBC;  Play-LM;  To learn control from play, we introduce Play-LMP, a selfsupervised method that learns to organize play behaviors in a latent space, then reuse them at test time to achieve specific goals.

  🔹 [Reward-Conditioned Policies](https://arxiv.org/pdf/1912.13465.pdf) :+1: :fire: :volcano:  ​

  Non-expert trajectories collected from suboptimal policies can be viewed as optimal supervision, not for maximizing the reward, but for matching the reward of the given trajectory.  Any experience collected by an agent can be used as optimal supervision when conditioned on the quality of a policy.

🔹 [Training Agents using Upside-Down Reinforcement Learning](https://arxiv.org/pdf/1912.02877.pdf) :fire:

UDRL: The goal of learning is no longer to maximize returns in expectation, but to learn to follow commands that may take various forms such as “achieve total reward R in next T time steps” or “reach state S in fewer than T time steps”.

🔹 [DEEP IMITATIVE MODELS FOR FLEXIBLE INFERENCE, PLANNING, AND CONTROL](https://arxiv.org/pdf/1810.06544.pdf) :+1: :fire:

We propose “Imitative Models” to combine the benefits of IL and goal-directed planning. Imitative Models are probabilistic predictive models of desirable behavior able to plan interpretable expert-like trajectories to achieve specified goals.

🔹 [ViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints](https://arxiv.org/pdf/2202.11271.pdf)

🔹 [Simplifying Deep Reinforcement Learning via Self-Supervision](https://arxiv.org/pdf/2106.05526.pdf) :no_mouth:

SSRL:  We demonstrate that, without policy gradient or value estimation, an iterative procedure of “labeling” data and supervised regression is sufficient to drive stable policy improvement.

  🔹 [Search on the Replay Buffer: Bridging Planning and Reinforcement Learning](https://arxiv.org/pdf/1906.05253.pdf) :fire: :+1:  ​ ​

  combines the strengths of planning and reinforcement learning

  🔹 [Phasic Self-Imitative Reduction for Sparse-Reward Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2206.12030.pdf) :fire: 🌋 

  PAIR: In the online phase, we perform RL training and collect rollout data while in the offline phase, we perform SL on those successful trajectories from the dataset. Task reduction.

  🔹 [SOLVING COMPOSITIONAL REINFORCEMENT LEARNING PROBLEMS VIA TASK REDUCTION](https://arxiv.org/pdf/2103.07607.pdf) :fire: 🌋 

  SIR: Task reduction tackles a hard-to-solve task by actively reducing it to an easier task whose solution is known by the RL agent.

  🔹 [DYNAMICAL DISTANCE LEARNING FOR SEMI-SUPERVISED AND UNSUPERVISED SKILL DISCOVERY](https://arxiv.org/pdf/1907.08225.pdf)

  dynamical distances: a measure of the expected number of time steps to reach a given goal state from any other states

  🔹 [Contextual Imagined Goals for Self-Supervised Robotic Learning](http://proceedings.mlr.press/v100/nair20a/nair20a.pdf) :+1: ​​  ​ ​  

  using the context-conditioned generative model to set goals that are appropriate to the current scene.

  🔹 [Reverse Curriculum Generation for Reinforcement Learning](https://arxiv.org/pdf/1707.05300.pdf) :+1: :fire:  ​

  **Finding the optimal start-state distribution**. Our method automatically generates a curriculum of start states that adapts to the agent’s performance, leading to efficient training on goal-oriented tasks.

  🔹 [Goal-Aware Prediction: Learning to Model What Matters](https://proceedings.icml.cc/static/paper_files/icml/2020/2981-Paper.pdf) :+1: :fire: :boom: Introduction is good!

  we propose to direct prediction towards task relevant information, enabling the model to be aware of the current task and **encouraging it to only model relevant quantities of the state space**, resulting in **a learning objective that more closely matches the downstream task**.  

  🔹 [C-LEARNING: LEARNING TO ACHIEVE GOALS VIA RECURSIVE CLASSIFICATION](https://openreview.net/pdf?id=tc5qisoB-C) :+1: :sweat_drops: :volcano: :boom:

  This Q-function is not useful for predicting or controlling the future state distribution. Fundamentally, this problem arises because the relationship between the reward function, the Q function, and the future state distribution in prior work remains unclear.  :ghost: [DIAYN?]

  on-policy ---> off-policy ---> goal-conditioned.

  🔹 [LEARNING TO UNDERSTAND GOAL SPECIFICATIONS BY MODELLING REWARD](https://openreview.net/pdf?id=H1xsSjC9Ym) :+1: :+1:  ​

  *ADVERSARIAL GOAL-INDUCED LEARNING FROM EXAMPLES*

  A framework within which instruction-conditional RL agents are trained using rewards obtained not from the environment, but from reward models which are jointly trained from expert examples.

  🔹 [Intrinsically Motivated Goal-Conditioned Reinforcement Learning: a Short Survey](https://arxiv.org/pdf/2012.09830.pdf) :volcano: :volcano: :droplet:

  This paper proposes a typology of these methods **[intrinsically motivated processes (IMP) (knowledge-based IMG + competence-based IMP); goal-conditioned RL agents]** at the intersection of deep rl and developmental approaches, surveys recent approaches and discusses future avenues.

  SEE: Language as a Cognitive Tool to Imagine Goals in Curiosity-Driven Exploration  

  🔹 [Self-supervised Learning of Distance Functions for Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/1907.02998.pdf)

  🔹 [PARROT: DATA-DRIVEN BEHAVIORAL PRIORS FOR REINFORCEMENT LEARNING](https://openreview.net/pdf?id=Ysuv-WOFeKR) :+1:

  We propose a method for pre-training behavioral priors that can capture complex input-output relationships observed in successful trials from a wide range of previously seen tasks.

   :ghost: see model-based <a href="#anchor-modelbasedddl">ddl</a>  

  🔹 [LEARNING WHAT TO DO BY SIMULATING THE PAST](https://openreview.net/pdf?id=kBVJ2NtiY-) :+1:  ​

  we propose the Deep Reward Learning by Simulating the Past (Deep RLSP) algorithm.

  🔹 [Weakly-Supervised Reinforcement Learning for Controllable Behavior](https://arxiv.org/pdf/2004.02860.pdf) :+1:

  two phase approach that learns a disentangled representation, and then uses it to guide exploration, propose goals, and inform a distance metric.

  🔹 [Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classification](https://arxiv.org/pdf/2103.12656.pdf) :volcano: :boom:

  we derive a method based on recursive classification that eschews auxiliary reward functions and instead directly learns a value function from transitions and successful outcomes.

  🔹 [C-learning: Learning to achieve goals via recursive classification]

  🔹 [Example-Based Offline Reinforcement Learning without Rewards](https://offline-rl-neurips.github.io/2021/pdf/53.pdf)

  🔹 [Outcome-Driven Reinforcement Learning via Variational Inference](https://arxiv.org/pdf/2104.10190.pdf) :+1: :fire: :droplet:

  by framing the problem of achieving desired outcomes as variational inference, we can derive an off-policy RL algorithm, a reward function learnable from environment interactions, and a novel Bellman backup that contains a state–action dependent dynamic discount factor for the reward and bootstrap.

  🔹 [Discovering Diverse Solutions in Deep Reinforcement Learning](https://arxiv.org/pdf/2103.07084.pdf) :+1:  ​

  learn infinitely many solutions by training a policy conditioned on a continuous or discrete low-dimensional latent variable.

  🔹 [Goal-Conditioned Reinforcement Learning with Imagined Subgoals](https://arxiv.org/pdf/2107.00541.pdf) :+1: :fire: :volcano:  ​ ​ ​

  This high-level policy predicts intermediate states halfway to the goal using the value function as a reachability metric. We don’t require the policy to reach these subgoals explicitly. Instead, we use them to define a prior policy, and incorporate this prior into a KL-constrained pi scheme to speed up and reg.

  🔹 [Goal-Space Planning with Subgoal Models](https://arxiv.org/pdf/2206.02902.pdf) :+1: :fire: 

  Goal-Space Planning (GSP): The key idea is to plan in a much smaller space of subgoals, and use these (high-level) subgoal values to update state values using subgoal-conditioned mode.

  🔹 [Goal-Space Planning with Subgoal Models](https://arxiv.org/pdf/2206.02902.pdf) :fire: 🌋  

  constraining background planning to a set of (abstract) subgoals and learning only local, subgoal-conditioned models. 

  🔹 [Discovering Generalizable Skills via Automated Generation of Diverse Tasks](https://arxiv.org/pdf/2106.13935.pdf) :no_mouth:

  As opposed to prior work on unsupervised discovery of skills which incentivizes the skills to produce different outcomes in the same environment, our method pairs each skill with a unique task produced by a trainable task generator. Procedural content generation (PCG).

🔹 [Unbiased Methods for Multi-Goal RL](https://arxiv.org/pdf/2106.08863.pdf) :confused:  :+1: :droplet:  

  First, we vindicate HER by proving that it is actually unbiased in deterministic environments, such as many optimal control settings. Next, for stochastic environments in continuous spaces, we tackle sparse rewards by directly taking the infinitely sparse reward limit.

🔹 [Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning](https://arxiv.org/pdf/2110.12985.pdf) :+1:

GACE: that can be utilized in a self-supervised way using auto-labeled goal states alongside reinforcement learning.

🔹 [DisCo RL: Distribution-Conditioned Reinforcement Learning for General-Purpose Policies](https://arxiv.org/pdf/2104.11707.pdf) :fire:  

Contextual policies provide this capability in principle, but the representation of the context determines the degree of generalization and expressivity. Categorical contexts preclude generalization to entirely new tasks. Goal-conditioned policies may enable some generalization, but cannot capture all tasks that might be desired.

🔹 [Demonstration-Conditioned Reinforcement Learning for Few-Shot Imitation](http://proceedings.mlr.press/v139/dance21a/dance21a.pdf) :fire:  

Given a training set consisting of demonstrations, reward functions and transition distributions for multiple tasks, the idea is to define a policy that takes demonstrations and current state as inputs, and to train this policy to maximize the average of the cumulative reward over the set of training tasks.

🔹 [C-LEARNING: HORIZON-AWARE CUMULATIVE ACCESSIBILITY ESTIMATION](https://arxiv.org/pdf/2011.12363.pdf) :confused: :droplet:

we introduce the concept of cumulative accessibility functions, which measure the reachability of a goal from a given state within a specified horizon.  

🔹 [C-PLANNING: AN AUTOMATIC CURRICULUM FOR LEARNING GOAL-REACHING TASKS](https://arxiv.org/pdf/2110.12080.pdf) :fire:

Frame the learning of the goal-conditioned policies as expectation maximization: the E-step corresponds to planning an optimal sequence of waypoints using graph search, while the M-step aims to learn a goal-conditioned policy to reach those waypoints.

🔹 [Bisimulation Makes Analogies in Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2204.13060.pdf) :fire: 🌋 

 We propose a new form of state abstraction called goal-conditioned bisimulation that captures functional equivariance, allowing for the reuse of skills to achieve new goals.

++DATA++

🔹 [Connecting the Dots Between MLE and RL for Sequence Prediction](https://arxiv.org/pdf/1811.09740.pdf) :+1:  

A rich set of other algorithms such as RAML, SPG, and data noising, have also been developed from different perspectives. This paper establishes a formal connection between these algorithms. We present a generalized entropy regularized policy optimization formulation, and show that the apparently distinct algorithms can all be reformulated as special instances of the framework, with the only difference being the configurations of a reward function and a couple of hyperparameters.

🔹 [Learning Data Manipulation for Augmentation and Weighting](https://openreview.net/pdf?id=BylmRrHg8S) :+1: :fire:  

We have developed a new method of learning different data manipulation schemes with the same single algorithm. Different manipulation schemes reduce to just different parameterization of the data reward function. The manipulation parameters are trained jointly with the target model parameters. (*Equivalence between Data and Reward*, *Gradient-based Reward Learning*)  

## Goal-relabeling & Self-imitation

  🔹 [Rewriting History with Inverse RL: Hindsight Inference for Policy Improvement](https://arxiv.org/pdf/2002.11089v1.pdf)

 HIPI: MaxEnt RL and MaxEnt inverse RL optimize **the same multi-task RL objective** with respect to trajectories and tasks, respectively.

🔹 [HINDSIGHT FORESIGHT RELABELING FOR META-REINFORCEMENT LEARNING](https://arxiv.org/pdf/2109.09031.pdf) :fire: :volcano:

Hindsight Foresight Relabeling (HFR): We construct a relabeling distribution using the combination of hindsight, which is used to relabel trajectories using reward functions from the training task distribution, and foresight, which takes the relabeled trajectories and computes the utility of each trajectory for each task.

🔹 [Generalized Hindsight for Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/57e5cb96e22546001f1d6520ff11d9ba-Paper.pdf) :+1: 

 Generalized Hindsight: an approximate inverse reinforcement learning technique for relabeling behaviors with the right tasks. 

🔹 [GENERALIZED DECISION TRANSFORMER FOR OFFLINE HINDSIGHT INFORMATION MATCHING](https://arxiv.org/pdf/2111.10364.pdf) :+1: :fire:

 We present Generalized Decision Transformer (GDT) for solving any HIM (hindsight information matching) problem, and show how different choices for the feature function and the anti-causal aggregator not only recover DT as a special case, but also lead to novel Categorical DT (CDT) and Bi-directional DT (BDT) for matching different statistics of the future.

  🔹 [Hindsight](https://zhuanlan.zhihu.com/p/191639584)    [Curriculum-guided Hindsight Experience Replay](https://papers.nips.cc/paper/9425-curriculum-guided-hindsight-experience-replay.pdf)    [COMPETITIVE EXPERIENCE REPLAY](https://arxiv.org/pdf/1902.00528.pdf)  :fire:   [Energy-Based Hindsight Experience Prioritization](https://arxiv.org/pdf/1810.01363.pdf)    [DHER: HINDSIGHT EXPERIENCE REPLAY FOR DYNAMIC GOALS](https://openreview.net/pdf?id=Byf5-30qFX)

  🔹 [Diversity-based Trajectory and Goal Selection with Hindsight Experience Replay](https://arxiv.org/pdf/2108.07887.pdf) :+1: 

   DTGSH: 1) a diversity-based trajectory selection module to sample valuable trajectories for the further goal selection; 2) a diversity-based goal selection module to select transitions with diverse goal states from the previously selected trajectories.

  🔹 [Exploration via Hindsight Goal Generation](http://papers.nips.cc/paper/9502-exploration-via-hindsight-goal-generation.pdf) :+1:  :fire:  ​

  a novel algorithmic framework that generates valuable hindsight goals which are easy for an agent to achieve in the short term and are also potential for guiding the agent to reach the actual goal in the long term.  

  🔹 [CURIOUS: Intrinsically Motivated Modular Multi-Goal Reinforcement Learning](https://arxiv.org/pdf/1810.06284.pdf) :+1:  ​

  This paper proposes CURIOUS, an algorithm that leverages 1) a modular Universal Value Function Approximator with hindsight learning to achieve a diversity of goals of different kinds within a unique policy and 2) an automated curriculum learning mechanism that biases the attention of the agent towards goals maximizing the absolute learning progress.  

  🔹 [Hindsight Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1903.07854.pdf) :fire:

  achieving imitation learning satisfying no need of demonstrations.  [see self-imitation learning]

  🔹 [MHER: Model-based Hindsight Experience Replay](https://arxiv.org/pdf/2107.00306.pdf) :+1:

  Replacing original goals with virtual goals generated from interaction with a trained dynamics model.

  🔹 [Policy Continuation with Hindsight Inverse Dynamics](https://papers.nips.cc/paper/2019/file/3891b14b5d8cce2fdd8dcdb4ded28f6d-Paper.pdf) :+1: :fire:  ​ ​

  This approach learns from Hindsight Inverse Dynamics based on Hindsight Experience Replay.  

  🔹 [USHER: Unbiased Sampling for Hindsight Experience Replay](https://arxiv.org/pdf/2207.01115.pdf) :fire: 🌋 

  We propose an asymptotically unbiased importance-sampling-based algorithm to address this problem without sacrificing performance on deterministic environments.

🔹 [Experience Replay Optimization](https://arxiv.org/pdf/1906.08387.pdf) :+1: :fire:

Self-imitation; experience replay: we propose a novel experience replay optimization (ERO) framework which alternately updates two policies: the agent policy, and the replay policy. The agent is updated to maximize the cumulative reward based on the replayed data, while the replay policy is updated to provide the agent with the most useful experiences.

🔹 [MODEL-AUGMENTED PRIORITIZED EXPERIENCE REPLAY](https://openreview.net/pdf?id=WuEiafqdy9H) :no_mouth:

We propose a novel experience replay method, which we call model-augmented priority experience replay (MaPER), that employs new learnable features driven from components in model-based RL (MbRL) to calculate the scores on experiences.

🔹 [TOPOLOGICAL EXPERIENCE REPLAY](https://arxiv.org/pdf/2203.15845.pdf) :fire:

TER: If the data sampling strategy ignores the precision of Q-value estimate of the next state, it can lead to useless and often incorrect updates to the Q-values.

🔹 [Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing](https://arxiv.org/pdf/1807.02322.pdf) :+1:

Our key idea is to express the expected return objective as a weighted sum of two terms: an expectation over the high-reward trajectories inside a memory buffer, and a separate expectation over trajectories outside of the buffer.

🔹 [RETRIEVAL-AUGMENTED REINFORCEMENT LEARNING](https://arxiv.org/pdf/2202.08417.pdf) :+1:

We augment an RL agent with a retrieval process (parameterized as a neural network) that has direct access to a dataset of experiences. The retrieval process is trained to retrieve information from the dataset that may be useful in the current context.

🔹 [VARIATIONAL ORACLE GUIDING FOR REINFORCEMENT LEARNING](https://openreview.net/pdf?id=pjqqxepwoMy) :fire:

Variational latent oracle guiding (VLOG) : An important but under-explored aspect is how to leverage oracle observation (the information that is invisible during online decision making, but is available during offline training) to facilitate learning.  

🔹 [WISH YOU WERE HERE: HINDSIGHT GOAL SELECTION FOR LONG-HORIZON DEXTEROUS MANIPULATION](https://openreview.net/pdf?id=FKp8-pIRo3y) :no_mouth:

We extend hindsight relabelling mechanisms to guide exploration along task-specific distributions implied by a small set of successful demonstrations.

🔹 [Hindsight Task Relabelling: Experience Replay for Sparse Reward Meta-RL](https://arxiv.org/pdf/2112.00901.pdf) :no_mouth:

HTR: we present a formulation of hindsight relabeling for meta-RL, which relabels experience during meta-training to enable learning to learn entirely using sparse reward.

🔹 [Remember and Forget for Experience Replay](http://proceedings.mlr.press/v97/novati19a/novati19a.pdf) :+1:

ReF-ER (1) skips gradients computed from experiences that are too unlikely with the current policy and (2) regulates policy changes within a trust region of the replayed behaviors.

🔹 [BENCHMARKING SAMPLE SELECTION STRATEGIES FOR BATCH REINFORCEMENT LEARNING](https://openreview.net/pdf?id=WxBFVNbDUT6) :+1:

We compare six variants of PER (temporal-difference error, n-step return, self-imitation learning objective, pseudo-count, uncertainty, and likelihood) based on various heuristic priority metrics that focus on different aspects of the offline learning setting.

🔹 [An Equivalence between Loss Functions and Non-Uniform Sampling in Experience Replay](https://proceedings.neurips.cc/paper/2020/file/a3bf6e4db673b6449c2f7d13ee6ec9c0-Paper.pdf) :fire:

We show that any loss function evaluated with non-uniformly sampled data can be transformed into another uniformly sampled loss function with the same expected gradient.  

🔹 [Self-Imitation Learning via Generalized Lower Bound Q-learning](https://arxiv.org/pdf/2006.07442.pdf) :fire:

To provide a formal motivation for the potential performance gains provided by self-imitation learning, we show that n-step lower bound Q-learning achieves a trade-off between fixed point bias and contraction rate, drawing close connections to the popular uncorrected n-step Q-learning.

🔹 [Understanding Multi-Step Deep Reinforcement Learning: A Systematic Study of the DQN Target](https://arxiv.org/pdf/1901.07510.pdf) :no_mouth:

we combine the n-step action-value alg. Retrace, Q-learning, Tree Backup, Sarsa, and Q(σ) with an architecture analogous to DQN. It suggests that off-policy correction is not always necessary for learning from samples from the experience replay buffer.

🔹 [Adaptive Trade-Offs in Off-Policy Learning](https://arxiv.org/pdf/1910.07478.pdf) :fire:

We take a unifying view of this space of algorithms (off-policy learning algorithms ), and consider their trade-offs of three fundamental quantities: update variance, fixed-point bias, and contraction rate.

- Imitation Learning (See Upper)

  🔹 [To Follow or not to Follow: Selective Imitation Learning from Observations](https://arxiv.org/pdf/1912.07670.pdf) :+1:  ​

  imitating every step in the demonstration often becomes infeasible when the learner and its environment are different from the demonstration.

- reward function

  🔹 [QUANTIFYING DIFFERENCES IN REWARD FUNCTIONS](https://openreview.net/pdf?id=LwEQnp6CYev) :volcano: :sweat_drops:  ​ ​

  we introduce the Equivalent-Policy Invariant Comparison (EPIC) distance to quantify the difference between two reward functions directly, **without training a policy**. We prove EPIC is invariant on an equivalence class of reward functions that always induce the same optimal policy.

<a name="anchor-modelbasedrl"></a>  

## Model-based RL & world models

  🔹 [A SURVEY ON MODEL-BASED REINFORCEMENT LEARNING](https://arxiv.org/pdf/2206.09328.pdf) 



  🔹 [Learning Latent Dynamics for Planning from Pixels](http://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf) ​​ :sweat_drops: :sweat_drops:  ​

  🔹 [DREAM TO CONTROL: LEARNING BEHAVIORS BY LATENT IMAGINATION](https://arxiv.org/pdf/1912.01603.pdf) :sweat_drops:  ​

  🔹 [CONTRASTIVE LEARNING OF STRUCTURED WORLD MODELS](https://arxiv.org/pdf/1911.12247.pdf) :fire: :volcano:  ​ ​

  🔹 [Learning Predictive Models From Observation and Interaction](https://arxiv.org/pdf/1912.12773.pdf) :fire:  ​related work is good!

  By combining interaction and observation data, our model is able to learn to generate predictions for complex tasks and new environments without costly expert demonstrations.  

  🔹 [medium](https://jonathan-hui.medium.com/rl-model-based-reinforcement-learning-3c2b6f0aa323)   [Tutorial on Model-Based Methods in Reinforcement Learning (icml2020)](https://sites.google.com/view/mbrl-tutorial)  :sweat_drops:  ​

  [rail](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_9_model_based_rl.pdf)  [Model-Based Reinforcement Learning: Theory and Practice](https://bair.berkeley.edu/blog/2019/12/12/mbpo/) :sweat_drops: ​​  ​ ​

  🔹 [What can I do here? A Theory of Affordances in Reinforcement Learning](http://proceedings.mlr.press/v119/khetarpal20a/khetarpal20a.pdf) :+1: :droplet:

  “affordances” to describe the fact that certain states enable an agent to do certain actions, in the context of embodied agents. In this paper, we develop a theory of affordances for agents who learn and plan in Markov Decision Processes.

  🔹 [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/pdf/1906.08253.pdf) :fire: :volcano: :droplet: :boom:  ​

   MBPO: we study the role of model usage in policy optimization both theoretically and empirically.

  🔹 [Visual Foresight: Model-based deep reinforcement learning for vision-based robotic control](https://arxiv.org/pdf/1812.00568.pdf) :+1:

  We presented an algorithm that leverages self-supervision from visual prediction to learn a deep dynamics model on images, and show how it can be embedded into a planning framework to solve a variety of robotic control tasks.

  🔹 [LEARNING STATE REPRESENTATIONS VIA RETRACING IN REINFORCEMENT LEARNING](https://openreview.net/pdf?id=CLpxpXqqBV) :+1:

  CCWM: a self-supervised instantiation of “learning via retracing” for joint representation learning and generative model learning under the model-based RL setting.

  🔹 [Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization](https://arxiv.org/pdf/2006.03647.pdf) :fire:  :volcano: :boom:

  we propose a novel concept of deployment efficiency, measuring the number of distinct data-collection policies that are used during policy learning.  ​

  🔹 [Context-aware Dynamics Model for Generalization in Model-Based Reinforcement Learning](https://arxiv.org/pdf/2005.06800.pdf) :+1: :fire:  ​ ​

   The intuition is that the true context of the underlying MDP can be captured from recent experiences. learning a global model that can generalize across different dynamics is a challenging task. To tackle this problem, we decompose the task of learning a global dynamics model into two stages: (a) learning a context latent vector that captures the local dynamics, then (b) predicting the next state conditioned on it

🔹 [Trajectory-wise Multiple Choice Learning for Dynamics Generalization in Reinforcement Learning](https://arxiv.org/pdf/2010.13303.pdf) :+1:  ​

The main idea is updating the most accurate prediction head to specialize each head in certain environments with similar dynamics, i.e., clustering environments.

  🔹 [Optimism is All You Need: Model-Based Imitation Learning From Observation Alone](https://arxiv.org/pdf/2102.10769.pdf) :sweat_drops:  ​

  🔹 [PlanGAN: Model-based Planning With Sparse Rewards and Multiple Goals](https://proceedings.neurips.cc/paper/2020/file/6101903146e4bbf4999c449d78441606-Paper.pdf) :+1:

  train an ensemble of conditional generative models (GANs) to generate plausible trajectories that lead the agent from its current state towards a specified goal. We then combine these imagined trajectories into a novel planning algorithm in order to achieve the desired goal as efficiently as possible.  

  🔹 [MODEL-ENSEMBLE TRUST-REGION POLICY OPTIMIZATION](https://arxiv.org/pdf/1802.10592.pdf) :+1:  

  we propose to use an ensemble of models to maintain the model uncertainty and regularize the learning process.

  🔹 [Sample Efficient Reinforcement Learning via Model-Ensemble Exploration and Exploitation](https://arxiv.org/pdf/2107.01825.pdf) :no_mouth:

  MEEE, a model-ensemble method that consists of optimistic exploration and weighted exploitation.  

🔹 [Regularizing Model-Based Planning with Energy-Based Models](http://proceedings.mlr.press/v100/boney20a/boney20a.pdf) :+1: :fire:

We focus on planning with learned dynamics models and propose to regularize it using energy estimates of state transitions in the environment. ---> probabilistic ensembles with trajectory sampling (PETS), DAE regularization;

🔹 [Model-Based Planning with Energy-Based Models](https://arxiv.org/pdf/1909.06878.pdf) :fire:

We show that energy-based models (EBMs) are a promising class of models to use for model-based planning. EBMs naturally support inference of intermediate states given start and goal state distributions.  

🔹 [Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?](http://proceedings.mlr.press/v119/filos20a/filos20a.pdf) :+1:

RIP: Our method can detect and recover from some distribution shifts, reducing the overconfident and catastrophic extrapolations in OOD scenes.  

🔹 [Model-Based Reinforcement Learning via Latent-Space Collocation](http://proceedings.mlr.press/v139/rybkin21b/rybkin21b.pdf) :fire:

LatCo:  It is easier to solve long-horizon tasks by planning sequences of states rather than just actions, as the effects of actions greatly compound over time and are harder to optimize.

🔹 [Reinforcement Learning with Action-Free Pre-Training from Videos](https://arxiv.org/pdf/2203.13880.pdf) :no_mouth:

APV: we pre-train an action-free latent video prediction model, and then utilize the pre-trained representations for efficiently learning actionconditional world models on unseen environments.

🔹 [Regularizing Trajectory Optimization with Denoising Autoencoders](https://arxiv.org/pdf/1903.11981.pdf) :fire:

The idea is that we want to reward familiar trajectories and penalize unfamiliar ones because the model is likely to make larger errors for the unfamiliar ones.

🔹 [Bridging Imagination and Reality for Model-Based Deep Reinforcement Learning](https://arxiv.org/pdf/2010.12142.pdf)  :+1: :fire: :droplet:

BIRD: our basic idea is to leverage information from real trajectories to endow policy improvement on imaginations with awareness of discrepancy between imagination and reality.

🔹 [ON-POLICY MODEL ERRORS IN REINFORCEMENT LEARNING](https://openreview.net/pdf?id=81e1aeOt-sd) :+1:

We present on-policy corrections (OPC) that combines real world data and a learned model in order to get the best of both worlds. The core idea is to exploit the real world data for on policy predictions and use the learned model only to generalize to different actions.

🔹 [ALGORITHMIC FRAMEWORK FOR MODEL-BASED DEEP REINFORCEMENT LEARNING WITH THEORETICAL GUARANTEES](https://arxiv.org/pdf/1807.03858.pdf) :+1:  :volcano: :droplet: :sweat_drops: :fire:  

SLBO:  We design a meta-algorithm with a theoretical guarantee of monotone improvement to a local maximum of the expected reward. The meta-algorithm iteratively builds a lower bound of the expected reward based on the estimated dynamical model and sample trajectories, and then maximizes the lower bound jointly over the policy and the model.

🔹 [Model-Augmented Q-Learning](https://arxiv.org/pdf/2102.03866.pdf)  :no_mouth:

We propose to estimate not only the Q-values but also both the transition and the reward with a shared network. We further utilize the estimated reward from the model estimators for Q-learning, which promotes interaction between the estimators.  

🔹 [Monotonic Robust Policy Optimization with Model Discrepancy](http://proceedings.mlr.press/v139/jiang21c/jiang21c.pdf) :+1: :volcano: :boom:

We propose a robust policy optimization approach, named MRPO, for improving both the average and worst-case performance of policies. We theoretically derived a lower bound for the worst-case performance of a given policy over all environments, and formulated an optimization problem to optimize the policy and sampling distribution together, subject to constraints that bounded the update step in policy optimization and statistical distance between the worst and average case environments.

🔹 [Policy Gradient Method For Robust Reinforcement Learning](https://arxiv.org/pdf/2205.07344.pdf) 

[[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/93db85ed909c13838ff95ccfa94cebd9.png) 

🔹 [Trust the Model When It Is Confident: Masked Model-based Actor-Critic](https://arxiv.org/pdf/2010.04893.pdf) :+1: :volcano:

We derive a general performance bound for model-based RL and theoretically show that the divergence between the return in the model rollouts and that in the real environment can be reduced with restricted model usage.  

🔹 [MBDP: A Model-based Approach to Achieve both Robustness and Sample Efficiency via Double Dropout Planning](https://arxiv.org/pdf/2108.01295.pdf) :+1: :fire:

MBDP: Model-Based Double-dropout Planning (MBDP) consists of two kinds of dropout mechanisms, where the rollout-dropout aims to improve the robustness with a small cost of sample efficiency, while the model-dropout is designed to compensate for the lost efficiency at a slight expense of robustness.

🔹 [PILCO (probabilistic inference for learning control)](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf)   [Deep PILCO](http://mlg.eng.cam.ac.uk/yarin/PDFs/DeepPILCO.pdf)  :+1:  

🔹 [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/pdf/1805.12114.pdf) :+1:

Employing uncertainty-aware dynamics models: we propose a new algorithm called probabilistic ensembles with trajectory sampling (PETS) that combines uncertainty-aware deep network dynamics models with sampling-based uncertainty propagation.

🔹 [Plan Online, Learn Offline: Efficient Learning and Exploration via Model-Based Control](https://arxiv.org/pdf/1811.01848.pdf) :+1: :fire: 

POLO utilizes a global value function approximation scheme, a local trajectory optimization subroutine, and an optimistic exploration scheme. 

🔹 [Learning Off-Policy with Online Planning](https://arxiv.org/pdf/2008.10066.pdf) :fire: 🌋

LOOP: We provide a theoretical analysis of this method, suggesting a tradeoff between model errors and value function errors and empirically demonstrate this tradeoff to be beneficial in deep reinforcement learning. H-step

🔹 [Calibrated Model-Based Deep Reinforcement Learning](https://arxiv.org/pdf/1906.08312.pdf) :fire:

This paper explores which uncertainties are needed for model-based reinforcement learning and argues that good uncertainties must be calibrated, i.e. their probabilities should match empirical frequencies of predicted events.

🔹 [Model Imitation for Model-Based Reinforcement Learning](https://arxiv.org/pdf/1909.11821.pdf) :+1: :volcano:

We propose to learn the transition model by matching the distributions of multi-step rollouts sampled from the transition model and the real ones via WGAN. We theoretically show that matching the two can minimize the difference of cumulative rewards between the real transition and the learned one.

🔹 [Model-based Policy Optimization with Unsupervised Model Adaptation](https://arxiv.org/pdf/2010.09546.pdf) :fire:

We derive a lower bound of the expected return, which inspires a bound maximization algorithm by aligning the simulated and real data distributions. To this end, we propose a novel model-based rl framework AMPO, which introduces unsupervised model adaptation to minimize the integral probability metric (IPM) between feature distributions from real and simulated data.

🔹 [Bidirectional Model-based Policy Optimization](http://proceedings.mlr.press/v119/lai20b/lai20b.pdf) :+1: :fire: :fire:

We propose to additionally construct a backward dynamics model to reduce the reliance on accuracy in forward model predictions: Bidirectional Model-based Policy Optimization (BMPO) to utilize both the forward model and backward model to generate short branched rollouts for policy optimization.  

🔹 [Backward Imitation and Forward Reinforcement Learning via Bi-directional Model Rollouts](https://arxiv.org/pdf/2208.02434.pdf) :fire:

BIFRL: the agent treats backward rollout traces as expert demonstrations for the imitation of excellent behaviors, and then collects forward rollout transitions for policy reinforcement.

🔹 [Self-Consistent Models and Values](https://openreview.net/pdf?id=x2rdRAx3QF) :fire:  

We investigate a way of augmenting model-based RL, by additionally encouraging a learned model and value function to be jointly self-consistent.

🔹 [MODEL-AUGMENTED ACTOR-CRITIC: BACKPROPAGATING THROUGH PATHS](https://arxiv.org/pdf/2005.08068.pdf) :fire: :volcano:

MAAC: We exploit the fact that the learned simulator is differentiable and optimize the policy with the analytical gradient. The objective is theoretically analyzed in terms of the model and value error, and we derive a policy improvement expression with respect to those terms.

🔹 [How to Learn a Useful Critic? Model-based Action-Gradient-Estimator Policy Optimization](https://arxiv.org/pdf/2004.14309.pdf) :fire:

MAGE backpropagates through the learned dynamics to compute gradient targets in temporal difference learning, leading to a critic tailored for policy improvement.

🔹 [Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/pdf/1803.00101.pdf)

🔹 [Discriminator Augmented Model-Based Reinforcement Learning](https://arxiv.org/pdf/2103.12999.pdf) :+1: :volcano:  

Our approach trains a discriminative model to assess the quality of sampled transitions during planning, and upweight or downweight value estimates computed from high and low quality samples, respectively. We can learn biased dynamics models with advantageous properties, such as reduced value estimation variance during planning.

🔹 [Variational Model-based Policy Optimization](https://arxiv.org/pdf/2006.05443.pdf) :+1: :fire:  :volcano:  :droplet:  

Jointly learn and improve model and policy using a universal objective function: We propose model-based and model-free policy iteration (actor-critic) style algorithms for the E-step and show how the variational distribution learned by them can be used to optimize the M-step in a fully model-based fashion.

🔹 [Model-Based Reinforcement Learning via Imagination with Derived Memory](https://openreview.net/pdf?id=jeATherHHGj) :fire: 

IDM: It enables the agent to learn policy from enriched diverse imagination with prediction-reliability weight, thus improving sample efficiency and policy robustness

🔹 [MISMATCHED NO MORE: JOINT MODEL-POLICY OPTIMIZATION FOR MODEL-BASED RL](https://arxiv.org/pdf/2110.02758.pdf) :fire:  :fire:

We propose a single objective for jointly training the model and the policy, such that updates to either component increases a lower bound on expected return.

🔹 [Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/pdf/1809.05214.pdf) :fire: :volcano:

MB-MPO: Using an ensemble of learned dynamic models, MB-MPO meta-learns a policy that can quickly adapt to any model in the ensemble with one policy gradient step, which foregoes the strong reliance on accurate learned dynamics models.  

🔹 [A RELATIONAL INTERVENTION APPROACH FOR UNSUPERVISED DYNAMICS GENERALIZATION IN MODELBASED REINFORCEMENT LEARNING](https://arxiv.org/pdf/2206.04551.pdf) :+1: :fire: 

We propose an intervention module to identify the probability of two estimated factors belonging to the same environment, and a relational head to cluster those estimated Zˆs are from the same environments with high probability, thus reducing the redundant information unrelated to the environment. 

🔹 [Value-Aware Loss Function for Model-based Reinforcement Learning](http://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf) :volcano:

Estimating a generative model that minimizes a probabilistic loss, such as the log-loss, is an overkill because it does not take into account the underlying structure of decision problem and the RL algorithm that intends to solve it. We introduce a loss function that takes the structure of the value function into account.

🔹 [Iterative Value-Aware Model Learning](https://proceedings.neurips.cc/paper/2018/file/7a2347d96752880e3d58d72e9813cc14-Paper.pdf)  :volcano: :boom:  

Iterative VAML, that benefits from the structure of how the planning is performed (i.e., through approximate value iteration) to devise a simpler optimization problem.

🔹 [Configurable Markov Decision Processes](https://arxiv.org/pdf/1806.05415.pdf#:~:text=A%20Configurable%20Markov%20Decision%20Process,the%20model%20and%20policy%20spaces.) :volcano: :boom: :droplet:

In Conf-MDPs the environment dynamics can be partially modified to improve the performance of the learning agent.  

🔹 [Bridging Worlds in Reinforcement Learning with Model-Advantage](https://openreview.net/pdf?id=XBRYX4c_xFQ) :volcano:

we show relationships between the proposed model advantage and generalization in RL — using which we provide guarantees on the gap in performance of an agent in new environments.

🔹 [Model-Advantage Optimization for Model-Based Reinforcement Learning](https://arxiv.org/pdf/2106.14080.pdf) :+1: :fire: :droplet:

a novel value-aware objective that is an upper bound on the absolute performance difference of a policy across two models.  

🔹 [Policy-Aware Model Learning for Policy Gradient Methods](https://arxiv.org/pdf/2003.00030.pdf) :fire: :volcano:

Decision-Aware Model Learning: We focus on policy gradient planning algorithms and derive new loss functions for model learning that incorporate how the planner uses the model.

🔹 [Gradient-Aware Model-Based Policy Search](https://arxiv.org/pdf/1909.04115.pdf) :fire: :volcano:  

Beyond Maximum Likelihood Model Estimation in Model-based Policy Search [ppt](http://www.honours-programme.deib.polimi.it/2018-1/Deliverable1/CSE_DORO_presentation.pdf)

🔹 [Model-Based Reinforcement Learning with Value-Targeted Regression](http://proceedings.mlr.press/v119/ayoub20a/ayoub20a.pdf) :confused:

🔹 [Decision-Aware Model Learning for Actor-Critic Methods: When Theory Does Not Meet Practice](http://proceedings.mlr.press/v137/lovatto20a/lovatto20a.pdf) :no_mouth:

we show empirically that combining Actor-Critic and value-aware model learning can be quite difficult and that naive approaches such as maximum likelihood estimation often achieve superior performance with less computational cost.  

🔹 [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://arxiv.org/pdf/2011.03506.pdf) :volcano: :droplet:

We introduced the principle of value equivalence: two models are value equivalent with respect to a set of functions and a set of policies if they yield the same updates of the former on the latter. Value equivalence formalizes the notion that models should be tailored to their future use and provides a mechanism to incorporate such knowledge into the model learning process.

🔹 [Proper Value Equivalence](https://arxiv.org/pdf/2106.10316.pdf) :sweat_drops:

We start by generalizing the concept of VE to order-k counterparts defined with respect to k applications of the Bellman operator. This leads to a family of VE classes that increase in size as k → \inf. In the limit, all functions become value functions, and we have a special instantiation of VE which we call proper VE or simply PVE.

🔹 [Minimax Model Learning](http://proceedings.mlr.press/v130/voloshin21a/voloshin21a.pdf) :volcano:  :droplet:  :boom:

our approach allows for greater robustness under model misspecification or distribution shift induced by learning/evaluating policies that are distinct from the data-generating policy.

🔹 [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/pdf/2111.08550.pdf) :fire:

AutoMBPO: we aim to investigate how to appropriately schedule these hyperparameters, i.e., real data ratio, model training frequency, policy training iteration, and rollout length, to achieve optimal performance of Dyna-style MBRL algorithms.

🔹 [Live in the Moment: Learning Dynamics Model Adapted to Evolving Policy](https://arxiv.org/pdf/2207.12141.pdf) :+1: 

Policy-adaptation Model-based Actor-Critic (PMAC), which learns a policy-adapted dynamics model based on a policy-adaptation mechanism. This mechanism dynamically adjusts the historical policy mixture distribution to ensure the learned model can continually adapt to the state-action visitation distribution of the evolving policy.

:o: Zero-Order Trajectory Optimizers / Planning

🔹 [Sample-efficient Cross-Entropy Method for Real-time Planning](https://arxiv.org/pdf/2008.06389.pdf) :droplet:

i Cross-Entropy Method (CEM):

🔹 [Extracting Strong Policies for Robotics Tasks from Zero-Order Trajectory Optimizers](https://openreview.net/pdf?id=Nc3TJqbcl3) :droplet:

 Adaptive Policy EXtraction (APEX):

  :o: dynamic distance learning   <a name="anchor-modelbasedddl"></a>  

  🔹 [MODEL-BASED VISUAL PLANNING WITH SELF-SUPERVISED FUNCTIONAL DISTANCES](https://openreview.net/pdf?id=UcoXdfrORC) :+1:

   We present a self-supervised method for model-based visual goal reaching, which uses both a visual dynamics model as well as a dynamical distance function learned using model-free rl. Related work!

  :o: model-based offline

  🔹 [Representation Balancing MDPs for Off-Policy Policy Evaluation](https://arxiv.org/pdf/1805.09044.pdf) :droplet:  ​

  🔹 [REPRESENTATION BALANCING OFFLINE MODEL-BASED REINFORCEMENT LEARNING](https://openreview.net/pdf?id=QpNz8r_Ri2Y) :droplet:  ​

  🔹 [Skill-based Model-based Reinforcement Learning](https://arxiv.org/pdf/2207.07560.pdf) 👍

  SkiMo: that enables planning in the skill space using a skill dynamics model, which directly predicts the skill outcomes, rather than predicting all small details in the intermediate states, step by step.

<a name="anchor-trainingrl"></a>  

## Training RL & Just Fast & Embedding? & OPE(DICE)

  🔹 [Reinforcement Learning: Theory and Algorithms](https://rltheorybook.github.io/rltheorybook_AJKS.pdf) 🌋 💦 



  🔹 [Leave no Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning](https://arxiv.org/pdf/1711.06782.pdf) :no_mouth:  ​

  🔹 [Predictive Information Accelerates Learning in RL](https://arxiv.org/pdf/2007.12401.pdf) :+1:  ​

  We train Soft Actor-Critic (SAC) agents from pixels with an auxiliary task that learns a compressed representation of the predictive information of the RL environment dynamics using a contrastive version of the Conditional Entropy Bottleneck (CEB) objective.

  🔹 [Speeding up Reinforcement Learning with Learned Models](https://upcommons.upc.edu/bitstream/handle/2117/175740/143210.pdf) :sweat_drops:  ​

  🔹 [DYNAMICS-AWARE EMBEDDINGS](https://arxiv.org/pdf/1908.09357.pdf) :+1:  ​

  A forward prediction objective for simultaneously learning embeddings of states and action sequences.

  🔹 [DIVIDE-AND-CONQUER REINFORCEMENT LEARNING](https://arxiv.org/pdf/1711.09874.pdf) :+1:

  we develop a novel algorithm that instead partitions the initial state space into “slices”, and optimizes an ensemble of policies, each on a different slice.

  🔹  [Continual Learning of Control Primitives: Skill Discovery via Reset-Games](https://arxiv.org/pdf/2011.05286.pdf) :+1: :fire:  

  We do this by exploiting the insight that the need to “reset" an agent to a broad set of initial states for a learning task provides a natural setting to learn a diverse set of “reset-skills".  

  🔹 [DIFFERENTIABLE TRUST REGION LAYERS FOR DEEP REINFORCEMENT LEARNING](https://openreview.net/pdf?id=qYZD-AO1Vn) :+1: :droplet:

  We derive trust region projections based on the Kullback-Leibler divergence, the Wasserstein L2 distance, and the Frobenius norm for Gaussian distributions. Related work is good!

  🔹 [BENCHMARKS FOR DEEP OFF-POLICY EVALUATION](https://openreview.net/pdf?id=kWSeGEeHvF8)  :+1: :fire: :droplet:

  DOPE is designed to measure the performance of **OPE** methods by 1) evaluating on challenging control tasks with properties known to be difficult for OPE methods, but which occur in real-world scenarios, 2) evaluating across a range of policies with different values, to directly measure performance on policy evaluation, ranking and selection, and 3) evaluating in ideal and adversarial settings in terms of dataset coverage and support.  

🔹 [Universal Off-Policy Evaluation](https://arxiv.org/pdf/2104.12820.pdf) :confused:

We take the first steps towards a universal off-policy estimator (UnO) that estimates and bounds the entire distribution of returns, and then derives estimates and simultaneous bounds for all parameters of interest.

  🔹 [Trajectory-Based Off-Policy Deep Reinforcement Learning](https://arxiv.org/pdf/1905.05710.pdf) :+1: :fire: :droplet:  ​

  Incorporation of previous rollouts via importance sampling greatly improves data-efficiency, whilst stochastic optimization schemes facilitate the escape from local optima.  

🔹 [Off-Policy Policy Gradient with State Distribution Correction](https://arxiv.org/pdf/1904.08473.pdf) :droplet:  

  🔹 [DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections](https://arxiv.org/pdf/1906.04733.pdf) :fire: :volcano: :boom:

  Off-Policy Policy Evaluation (OPE) --->  Learning Stationary Distribution Corrections ---> Off-Policy Estimation with Multiple Unknown Behavior Policies. , DualDICE, for estimating the discounted stationary distribution corrections.

  🔹 [AlgaeDICE: Policy Gradient from Arbitrary Experience](https://arxiv.org/pdf/1912.02074.pdf) :+1: :volcano: :droplet:  ​ ​ ​

  We introduce a new formulation of max-return optimization that allows the problem to be re-expressed by an expectation over an arbitrary behavior-agnostic and off-policy data distribution. ALgorithm for policy Gradient from Arbitrary Experience via DICE (AlgaeDICE).

  🔹 [GENDICE: GENERALIZED OFFLINE ESTIMATION OF STATIONARY VALUES](https://arxiv.org/pdf/2002.09072.pdf) :+1: :fire: :fire: ​​ ​ ​

  Our approach is based on estimating a ratio that corrects for the discrepancy between the stationary and empirical distributions, derived from fundamental properties of the stationary distribution, and exploiting constraint reformulations based on variational divergence minimization.

  🔹 [GradientDICE: Rethinking Generalized Offline Estimation of Stationary Values](http://proceedings.mlr.press/v119/zhang20r/zhang20r.pdf) :confused:  ​

  🔹 [Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation](https://papers.nips.cc/paper/2018/file/dda04f9d634145a9c68d5dfe53b21272-Paper.pdf) :+1: :fire: :droplet:  ​

  The key idea is to apply importance sampling on the average visitation distribution of single steps of state-action pairs, instead of the much higher dimensional distribution of whole trajectories.

  🔹 [Off-Policy Evaluation via the Regularized Lagrangian](https://arxiv.org/pdf/2007.03438.pdf) :fire: :confused: :droplet:  ​

  we unify these estimators (DICE) as regularized Lagrangians of the same linear program.

🔹 [OptiDICE: Offline Policy Optimization via Stationary Distribution Correction Estimation](https://arxiv.org/pdf/2106.10783.pdf) :+1: :volcano:

Our algorithm, OptiDICE, directly estimates the stationary distribution corrections of the optimal policy and does not rely on policy-gradients, unlike previous offline RL algorithms.

🔹 [SMODICE: Versatile Offline Imitation Learning via State Occupancy Matching](https://arxiv.org/pdf/2202.02433.pdf) 

🔹 [DEMODICE: OFFLINE IMITATION LEARNING WITH SUPPLEMENTARY IMPERFECT DEMONSTRATIONS](https://openreview.net/pdf?id=BrPdX1bDZkQ) 🌋 :fire: 👍

An algorithm for offline IL from expert and imperfect demonstrations that achieves state-of-the-art performance on various offline IL tasks.

🔹 [OFF-POLICY CORRECTION FOR ACTOR-CRITIC ALGORITHMS IN DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/2208.00755.pdf) 😶 

AC-Off-POC: Through a novel discrepancy measure computed by the agent’s most recent action decisions on the states of the randomly sampled batch of transitions, the approach does not require actual or estimated action probabilities for any policy and offers an adequate one-step importance sampling.

🔹 [A Deep Reinforcement Learning Approach to Marginalized Importance Sampling with the Successor Representation](http://proceedings.mlr.press/v139/fujimoto21a/fujimoto21a.pdf) :fire: :fire: 

We bridge the gap between MIS and deep RL  by observing that the density ratio can be computed from the successor representation of the target policy. The successor representation can be trained through deep RL methodology and decouples the reward optimization from the dynamics of the environment, making the resulting algorithm stable and applicable to high-dimensional domains.

🔹 [How Far I’ll Go: Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression](https://arxiv.org/pdf/2206.03023.pdf) 🌋

Goal-conditioned f-Advantage Regression (GoFAR), a novel regressionbased offline GCRL algorithm derived from a state-occupancy matching perspective; the key intuition is that the goal-reaching task can be formulated as a stateoccupancy matching problem between a dynamics-abiding imitator agent and an expert agent that directly teleports to the goal.

  🔹 [Minimax Weight and Q-Function Learning for Off-Policy Evaluation](http://proceedings.mlr.press/v119/uehara20a/uehara20a.pdf) :fire: :droplet:  ​ ​

  Minimax Weight Learning (MWL); Minimax Q-Function Learning. Doubly Robust Extension and Sample Complexity of MWL & MQL.

  🔹 [Minimax Value Interval for Off-Policy Evaluation and Policy Optimization](https://arxiv.org/pdf/2002.02081.pdf) :+1: :fire: :volcano: :droplet:

  we derive the minimax value intervals by slightly altering the derivation of two recent methods [1], one of “weight-learning” style (Sec. 4.1) and one of “value-learning” style (Sec. 4.2), and show that under certain conditions, they merge into a single unified value interval whose validity only relies on either Q or W being well-specified (Sec. 4.3).  

  🔹 [Reinforcement Learning via Fenchel-Rockafellar Duality](https://arxiv.org/pdf/2001.01866.pdf) :fire: :volcano: :droplet:  ​ ​ ​

  *Policy Evaluation*: LP form of Q ---> policy evaluation via largrangian ---> change the problem before applying duality (constant function, f-divergence, fenchel-rockafellar duality);  *Policy Optimization*: policy gradient ---> offline policy gradient via the lagrangian ---> fenchel-rockafellar duality for the regularized optimization (regularization with the kl-d) ---> imitation learning;  *RL with the LP form of V*: max-likelihood policy learning ---> policy evaluation with the V-lp; *Undiscounted Settings*

  🔹 [ADVANTAGE-WEIGHTED REGRESSION: SIMPLE AND SCALABLE OFF-POLICY REINFORCEMENT LEARNING](https://arxiv.org/pdf/1910.00177.pdf) :+1: :fire:  :boom:

  Our proposed approach, which we refer to as advantage-weighted regression (AWR), consists of two standard supervised learning steps: one to regress onto target values for a value function, and another to regress onto weighted target actions for the policy.  [see MPO]

🔹 [Relative Entropy Policy Search](https://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewFile/1851/2264) :fire: ​

REPS:  it allows an exact policy update and may use data generated while following an unknown policy to generate a new, better policy.

  🔹 [Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/pdf/1709.10089.pdf) :fire:

  We present a system to utilize demonstrations along with reinforcement learning to solve complicated multi-step tasks. Q-Filter. BC.

  🔹 [Fitted Q-iteration by Advantage Weighted Regression](https://papers.nips.cc/paper/2008/file/f79921bbae40a577928b76d2fc3edc2a-Paper.pdf) :+1:  :fire:  :fire:  ​

  we show that by using a soft-greedy action selection the policy improvement step used in FQI can be simplified to an inexpensive advantage weighted regression. <--- greedy action selection in continuous.

  🔹 [Q-Value Weighted Regression: Reinforcement Learning with Limited Data](https://arxiv.org/pdf/2102.06782.pdf) :fire:  :volcano:

  QWR: We replace the value function critic of AWR with a Q-value function.  AWR --> QWR.

🔹 [SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning](http://proceedings.mlr.press/v139/lee21g/lee21g.pdf) :fire:  :+1:

SUNRISE integrates two key ingredients: (a) ensemble-based weighted Bellman backups, which re-weight target Q-values based on uncertainty estimates from a Q-ensemble, and (b) an inference method that selects actions using the highest upper-confidence bounds for efficient exploration. [Rainbow]

🔹 [Revisiting Rainbow: Promoting more Insightful and Inclusive Deep Reinforcement Learning Research](http://proceedings.mlr.press/v139/ceron21a/ceron21a.pdf) :volcano:

🔹 [Explaining Off-Policy Actor-Critic From A Bias-Variance Perspective](https://arxiv.org/pdf/2110.02421.pdf) :confused:

To understand an off-policy actor-critic algorithm, we show the policy evaluation error on the expected distribution of transitions decomposes into the Bellman error, the bias from policy mismatch, and the variance from sampling.  

🔹 [SOPE: Spectrum of Off-Policy Estimators](https://openreview.net/pdf?id=Mfi0LZmFB5a) :+1: :fire: :volcano:

Combining Trajectory-Based and Density-Based Importance Sampling: We present a new perspective in off-policy evaluation connecting two popular estimators, PDIS and SIS, and show that PDIS and SIS lie as endpoints on the Spectrum of Off-Policy Estimators SOPEn which interpolates between them.

🔹 [Finite-Sample Analysis of Off-Policy TD-Learning via Generalized Bellman Operators](https://openreview.net/pdf?id=esCx4oejjxw) :fire: :volcano: :droplet:

Generalized Bellman Operator: Qπ (λ), Tree-Backup(λ) (henceforth denoted by TB(λ)), Retrace(λ), and Q-trace.

🔹 [Efficient Continuous Control with Double Actors and Regularized Critics](https://arxiv.org/pdf/2106.03050.pdf) :fire:

DARC: We show that double actors help relieve overestimation bias in DDPG if built upon single critic, and underestimation bias in TD3 if built upon double critics. (they enhance the exploration ability of the agent.)

🔹 [A Unified Off-Policy Evaluation Approach for General Value Function](https://arxiv.org/pdf/2107.02711.pdf) :droplet:

GenTD:

Model Selection:

🔹 [Pessimistic Model Selection for Offline Deep Reinforcement Learning](https://arxiv.org/pdf/2111.14346.pdf) :fire: :droplet:

We propose a pessimistic model selection (PMS) approach for offline DRL with a theoretical guarantee, which features a provably effective framework for finding the best policy among a set of candidate models.  

  🔹 [PARAMETER-BASED VALUE FUNCTIONS](https://openreview.net/pdf?id=tV6oBfuyLTQ) :+1:  ​

  Parameter-Based Value Functions (PBVFs) whose inputs include the policy parameters.

  🔹 [Reinforcement Learning without Ground-Truth State](https://arxiv.org/pdf/1905.07866.pdf) :+1:

  relabeling the original goal with the achieved goal to obtain positive rewards  

  🔹 [Ecological Reinforcement Learning](https://arxiv.org/pdf/2006.12478.pdf) :+1:

  🔹 [Control Frequency Adaptation via Action Persistence in Batch Reinforcement Learning](https://icml.cc/virtual/2020/poster/6146) :+1:   ​

  🔹 [Taylor Expansion Policy Optimization](http://proceedings.mlr.press/v119/tang20d/tang20d.pdf) :+1: :boom: :volcano: :droplet:  ​

  a policy optimization formalism that generalizes prior work (e.g., TRPO) as a firstorder special case. We also show that Taylor expansions intimately relate to off-policy evaluation.

  🔹 [Policy Information Capacity: Information-Theoretic Measure for Task Complexity in Deep Reinforcement Learning](https://arxiv.org/pdf/2103.12726.pdf) :+1:  ​

  Policy Information Capacity: Information-Theoretic Measure for Task Complexity in Deep Reinforcement Learning.

  🔹 [Deep Reinforcement Learning with Robust and Smooth Policy](http://proceedings.mlr.press/v119/shen20b/shen20b.pdf) :+1:  

  Motivated by the fact that many environments with continuous state space have smooth transitions, we propose to learn a smooth policy that behaves smoothly with respect to states. We develop a new framework — Smooth Regularized Reinforcement Learning (SR2L), where the policy is trained with smoothness-inducing regularization.

  🔹 [If MaxEnt RL is the Answer, What is the Question?](https://arxiv.org/pdf/1910.01913.pdf) :+1:  :fire: :volcano:

  🔹 [Maximum Entropy RL (Provably) Solves Some Robust RL Problems](https://arxiv.org/pdf/2103.06257.pdf) :fire: :volcano:

Our main contribution is a set of proofs showing that standard MaxEnt RL optimizes lower bounds on several possible robust objectives, reflecting a degree of robustness to changes in the dynamics and to certain changes in the reward.

🔹 [Your Policy Regularizer is Secretly an Adversary](https://arxiv.org/pdf/2203.12592.pdf) :sweat_drops:

🔹 [Estimating Q(s, s') with Deep Deterministic Dynamics Gradients](https://arxiv.org/pdf/2002.09505.pdf) :+1: :fire:  

 We highlight the benefits of this approach in terms of value function transfer, learning within redundant action spaces, and learning off-policy from state observations generated by sub-optimal or completely random policies.

🔹 [RANDOMIZED ENSEMBLED DOUBLE Q-LEARNING: LEARNING FAST WITHOUT A MODEL](https://arxiv.org/pdf/2101.05982.pdf) :fire:

REDQ: (i) a Update-To-Data (UTD) ratio >> 1; (ii) an ensemble of Q functions; (iii) in-target minimization across a random subset of Q functions from the ensemble.

🔹 [DROPOUT Q-FUNCTIONS FOR DOUBLY EFFICIENT REINFORCEMENT LEARNING](https://openreview.net/pdf?id=xCVJMsPv3RT) :no_mouth:

To make REDQ more computationally efficient, we propose a method of improving computational efficiency called Dr.Q, which is a variant of REDQ that uses a small ensemble of dropout Q-functions.

🔹 [Disentangling Dynamics and Returns: Value Function Decomposition with Future Prediction](https://arxiv.org/pdf/1905.11100.pdf) :+1:  ​

we propose a two-step understanding of value estimation from the perspective of future prediction, through decomposing the value function into a reward-independent future dynamics part and a policy-independent trajectory return part.

🔹 [DisCor: Corrective Feedback in Reinforcement Learning via Distribution Correction](https://arxiv.org/pdf/2003.07305.pdf)

🔹 [Regret Minimization Experience Replay in Off-Policy Reinforcement Learning](https://arxiv.org/pdf/2105.07253.pdf) :fire: :volcano:

ReMERN and ReMERT: We start from the regret minimization objective, and obtain an optimal prioritization strategy for Bellman update that can directly maximize the return of the policy. The theory suggests that data with higher hindsight TD error, better on-policiness and more accurate Q value should be assigned with higher weights during sampling.

🔹 [Off-Policy Policy Gradient Algorithms by Constraining the State Distribution Shift](https://arxiv.org/pdf/1911.06970.pdf) :+1:

Existing off-policy gradient based methods do not correct for the state distribution mismatch, and in this work we show that instead of computing the ratio over state distributions, we can instead minimize the KL between the target and behaviour state distributions to account for the state distribution shift in off-policy learning.

🔹 [Fast Efficient Hyperparameter Tuning for Policy Gradient Methods](https://openreview.net/pdf?id=r1exYESgLH) :no_mouth:

Hyperparameter Optimisation on the Fly (HOOF): The main idea is to use existing trajectories sampled by the policy grad. method to optimise a one-step improvement objective, yielding a sample and computationally efficient alg. that is easy to implement.

🔹 [REWARD SHIFTING FOR OPTIMISTIC EXPLORATION AND CONSERVATIVE EXPLOITATION](https://openreview.net/pdf?id=CNY9h3uyfiO) :no_mouth:

We bring the key insight that a positive reward shifting leads to conservative exploitation, while a negative reward shifting leads to curiosity-driven exploration.

🔹 [Heuristic-Guided Reinforcement Learning](https://arxiv.org/pdf/2106.02757.pdf) :fire: 🌋 

HuRL: We show how heuristic-guided RL induces a much shorter-horizon subproblem that provably solves the original task. Our framework can be viewed as a horizon-based regularization for controlling bias and variance in RL under a finite interaction budget.

🔹 [Using a Logarithmic Mapping to Enable Lower Discount Factors in Reinforcement Learning](https://arxiv.org/pdf/1906.00572.pdf) :fire: :droplet:

Our results provide strong evidence for our hypothesis that large differences in action-gap sizes are detrimental to the performance of approximate RL.  

🔹 [ORCHESTRATED VALUE MAPPING FOR REINFORCEMENT LEARNING](https://arxiv.org/pdf/2203.07171.pdf) :fire:

We present a general convergent class of reinforcement learning algorithms that is founded on two distinct principles: (1) mapping value estimates to a different space using arbitrary functions from a broad class, and (2) linearly decomposing the reward signal into multiple channels.  

🔹 [Discount Factor as a Regularizer in Reinforcement Learning](http://proceedings.mlr.press/v119/amit20a/amit20a.pdf) :fire: :+1:

We show an explicit equivalence between using a reduced discount factor and adding an explicit regularization term to the algorithm’s loss.  

🔹 [Learning to Score Behaviors for Guided Policy Optimization](https://arxiv.org/pdf/1906.04349.pdf) :fire: :droplet:

 We show that by utilizing the dual formulation of the WD, we can learn score functions over policy behaviors that can in turn be used to lead policy optimization towards (or away from) (un)desired behaviors.

🔹 [Dual Policy Distillation](https://arxiv.org/pdf/2006.04061.pdf) :volcano:

DPD: a student-student framework in which two learners operate on the same environment to explore different perspectives of the environment and extract knowledge from each other to enhance their learning.

🔹 [Jump-Start Reinforcement Learning](https://arxiv.org/pdf/2204.02372.pdf) :volcano:

JSRL: an algorithm that employs two policies to solve tasks: a guide-policy, and an exploration-policy. By using the guide-policy to form a curriculum of starting states for the exploration-policy, we are able to efficiently improve performance on a set of simulated robotic tasks.

🔹 [Distilling Policy Distillation](http://proceedings.mlr.press/v89/czarnecki19a/czarnecki19a.pdf) :volcano: :+1:

We sought to highlight some of the strengths, weaknesses, and potential mathematical inconsistencies in different variants of distillation used for policy knowledge transfer in reinforcement learning.

  🔹 [Regularized Policies are Reward Robust](http://proceedings.mlr.press/v130/husain21a/husain21a.pdf) :fire: :droplet:  ​

  we find that the optimal policy found by a regularized objective is precisely an optimal policy of a reinforcement learning problem under a worst-case adversarial reward.

  🔹 [Reinforcement Learning as One Big Sequence Modeling Problem](https://people.eecs.berkeley.edu/~janner/trajectory-transformer/files/trajectory-transformer.pdf) :+1: :fire: :droplet:  ​ ​ ​

  Addressing RL as a sequence modeling problem significantly simplifies a range of design decisions: we no longer require separate behavior policy constraints, as is common in prior work on offline model-free RL, and we no longer require ensembles or other epistemic uncertainty estimators, as is common in prior work on model-based RL.

  🔹 [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) :fire: 🌋 

  🔹 [Prompting Decision Transformer for Few-Shot Policy Generalization](https://arxiv.org/pdf/2206.13499.pdf) :fire: 🌋 

  We propose a Prompt-based Decision Transformer (Prompt-DT), which leverages the sequential modeling ability of the Transformer architecture and the prompt framework to achieve few-shot adaptation in offline RL.

  🔹 [Bootstrapped Transformer for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.08569.pdf) 😶 

  Bootstrapped Transformer, which incorporates the idea of bootstrapping and leverages the learned model to self-generate more offline data to further boost the sequence model training.

  🔹 [On-Policy Deep Reinforcement Learning for the Average-Reward Criterion](https://arxiv.org/pdf/2106.07329.pdf) :+1: :fire:

  By addressing the average-reward criterion directly, we then derive a novel bound which depends on the average divergence between the two policies and Kemeny’s constant.

  🔹 [Average-Reward Reinforcement Learning with Trust Region Methods](https://arxiv.org/pdf/2106.03442.pdf) :+1:  :fire:

  Firstly, we develop a unified trust region theory with discounted and average criteria. With the average criterion, a novel performance bound within the trust region is derived with the Perturbation Analysis (PA) theory. Secondly, we propose a practical algorithm named Average Policy Optimization (APO), which improves the value estimation with a novel technique named Average Value Constraint.

  🔹 [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) :+1:  :fire: :boom:  ​ ​

  🔹 [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/pdf/1604.06778.pdf) :+1: :fire:  ​ ​

  🔹 [P3O: Policy-on Policy-off Policy Optimization](http://proceedings.mlr.press/v115/fakoor20a/fakoor20a.pdf) :fire:

  This paper develops a simple alg. named P3O that interleaves offpolicy updates with on-policy updates.  

🔹 [Policy Gradients Incorporating the Future](https://arxiv.org/pdf/2108.02096.pdf) :fire:

we consider the problem of incorporating information from the entire trajectory in model-free online and offline RL algorithms, enabling an agent to use information about the future to accelerate and improve its learning.  

🔹 [Generalizable Episodic Memory for Deep Reinforcement Learning](https://arxiv.org/pdf/2103.06469.pdf) :+1: :fire:

Generalizable Episodic Memory: We propose Generalizable Episodic Memory (GEM), which effectively organizes the state-action values of episodic memory in a generalizable manner and supports implicit planning on memorized trajectories.

🔹 [Generalized Proximal Policy Optimization with Sample Reuse](https://openreview.net/pdf?id=in_RVSTqYxK) :+1: :fire: :volcano:

GePPO: We combine the theoretically supported stability benefits of on-policy algorithms with the sample efficiency of off-policy algorithms. We develop policy improvement guarantees that are suitable for the off-policy setting, and connect these bounds to the clipping mechanism used in Proximal Policy Optimization.

🔹 [Zeroth-Order Supervised Policy Improvement](https://arxiv.org/pdf/2006.06600.pdf) :fire: :fire:  ​

The policy learning of ZOSPI has two steps: 1), it samples actions and evaluates those actions with a learned value estimator, and 2) it learns to perform the action with the highest value through supervised learning.  

🔹 [SAMPLE EFFICIENT ACTOR-CRITIC WITH EXPERIENCE REPLAY](https://arxiv.org/pdf/1611.01224.pdf) :+1: :fire:

including truncated importance sampling with bias correction, stochastic dueling network architectures, and a new trust region policy optimization.

🔹 [Safe and efficient off-policy reinforcement learning](https://arxiv.org/pdf/1606.02647.pdf) :fire: :droplet:  ​ ​

Retrace(λ); low variance, safe, efficient,

🔹 [Relative Entropy Regularized Policy Iteration](https://arxiv.org/pdf/1812.02256.pdf) :fire: :fire:

The algorithm alternates between Q-value estimation, local policy improvement and parametric policy fitting; hard constraints control the rate of change of the policy. And a decoupled update for mean and covarinace of a Gaussian policy avoids premature convergence.  [see MPO]

🔹  [Q-Learning for Continuous Actions with Cross-Entropy Guided Policies](https://arxiv.org/pdf/1903.10605.pdf) :+1:

 Our approach trains the Q-function using iterative sampling with the Cross-Entropy Method (CEM), while training a policy network to imitate CEM’s sampling behavior.

🔹 [SUPERVISED POLICY UPDATE FOR DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1805.11706.pdf) :+1: :fire: :fire:  ​ ​ ​

FORWARD AGGREGATE AND DISAGGREGATE KL CONSTRAINTS; BACKWARD KL CONSTRAINT; L CONSTRAINT;

🔹 [Maximizing Ensemble Diversity in Deep Q-Learning](https://arxiv.org/pdf/2006.13823.pdf) :no_mouth:  

Reducing overestimation bias by increasing representation dissimilarity in ensemble based deep q-learning.

🔹 [Value-driven Hindsight Modelling](https://arxiv.org/pdf/2002.08329.pdf) :confused:

we propose to learn what to model in a way that can directly help value prediction.  

🔹 [Dual Policy Iteration](https://arxiv.org/pdf/1805.10755.pdf) :+1: :fire:

DPI: We present and analyze Dual Policy Iteration—a framework that alternatively computes a non-reactive policy via more advanced and systematic search, and updates a reactive policy via imitating the non-reactive one. [MPO, AWR]

🔹 [Regret Minimization for Partially Observable Deep Reinforcement Learning](http://proceedings.mlr.press/v80/jin18c/jin18c.pdf) :confused:

 ​

 🔹 [THE IMPORTANCE OF PESSIMISM IN FIXED-DATASET POLICY OPTIMIZATION](https://arxiv.org/pdf/2009.06799.pdf)  :volcano: :boom:  :ghost:  :sweat_drops:  ​

Algs can follow the pessimism principle, which states that we should choose the policy which acts optimally in the worst possible world. We show why pessimistic algorithms can achieve good performance even when the dataset is not informative of every policy, and derive families of algorithms which follow this principle.

🔹 [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/pdf/1702.08892.pdf) :fire: :boom:  :volcano:  ​

we develop a new RL algorithm, Path Consistency Learning (PCL), that minimizes a notion of soft consistency error along multi-step action sequences extracted from both on- and off-policy traces.

🔹 [Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/pdf/1704.06440.pdf) :+1:  :droplet:  ​

The soft Q-learning loss gradient can be interpreted as a policy gradient term plus a baseline-error-gradient term, corresponding to policy gradient instantiations such as A3C.

🔹 [An operator view of policy gradient methods](https://arxiv.org/pdf/2006.11266.pdf) :fire:

We use this framework to introduce operator-based versions of well-known policy gradient methods.

🔹 [MAXIMUM REWARD FORMULATION IN REINFORCEMENT LEARNING](https://arxiv.org/pdf/2010.03744.pdf) :droplet:

We formulate an objective function to maximize the expected maximum reward along a trajectory, derive a novel functional form of the Bellman equation, introduce the corresponding Bellman operators, and provide a proof of convergence.  

🔹 [Why Should I Trust You, Bellman? The Bellman Error is a Poor Replacement for Value Error](https://arxiv.org/pdf/2201.12417.pdf) :fire:

The magnitude of the Bellman error is smaller for biased value functions due to cancellations caused from both sides of the Bellman equation. The relationship between Bellman error and value error is broken if the dataset is missing relevant transitions.

🔹 [CONVERGENT AND EFFICIENT DEEP Q NETWORK ALGORITHM](https://openreview.net/pdf?id=OJm3HZuj4r7) :confused:

We show that DQN can indeed diverge and cease to operate in realistic settings. we propose a convergent DQN (C-DQN) that is guaranteed to converge.

🔹 [LEARNING SYNTHETIC ENVIRONMENTS AND REWARD NETWORKS FOR REINFORCEMENT LEARNING](https://arxiv.org/pdf/2202.02790.pdf) :+1:

We use bi-level optimization to evolve SEs and RNs: the inner loop trains the RL agent, and the outer loop trains the parameters of the SE / RN via an evolution strategy.  

🔹 [IS HIGH VARIANCE UNAVOIDABLE IN RL? A CASE STUDY IN CONTINUOUS CONTROL](https://openreview.net/pdf?id=9xhgmsNVHu) 

🔹 [Reinforcement Learning with a Terminator](https://arxiv.org/pdf/2205.15376.pdf) :fire: 🔥 

We define the Termination Markov Decision Process (TerMDP), an extension of the MDP framework, in which episodes may be interrupted by an external non-Markovian observer. 

🔹 [Truly Deterministic Policy Optimization](https://arxiv.org/pdf/2205.15379.pdf) :confused: :+1: 

We proposed a deterministic policy gradient method (TDPO) based on the use of a deterministic Vine (DeVine) gradient estimator and the Wasserstein metric. We proved monotonic payoff guarantees for our method, and defined a novel surrogate for policy optimization. 

🔹 [Automated Reinforcement Learning (AutoRL): A Survey and Open Problems](https://arxiv.org/pdf/2201.03916.pdf) 💦 

🔹 [CGAR: Critic Guided Action Redistribution in Reinforcement Leaning](https://arxiv.org/pdf/2206.11494.pdf) 😶

 the Q value predicted by the critic is a better signal to redistribute the action originally sampled from the policy distribution predicted by the actor. 

🔹 [Value Function Decomposition for Iterative Design of Reinforcement Learning Agents](https://arxiv.org/pdf/2206.13901.pdf) 😶

SAC-D: We also introduce decomposition-based tools that exploit this information, including a new reward influence metric, which measures each reward component’s effect on agent decision-making. 

🔹 [Emphatic Algorithms for Deep Reinforcement Learning](https://arxiv.org/pdf/2106.11779.pdf) :fire: 

🔹 [Off-Policy Evaluation for Large Action Spaces via Embeddings](https://arxiv.org/pdf/2202.06317.pdf) 

we propose a new OPE estimator that leverages marginalized importance weights when action embeddings provide structure in the action space. [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/6da9003b743b65f4c0ccd295cc484e57.png) 





🔹 [Gradient Temporal-Difference Learning with Regularized Corrections](https://arxiv.org/pdf/2007.00611.pdf) :fire: 

🔹 [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/pdf/2205.07802.pdf) :fire: 

"Your assumptions are your windows on the world. Scrub them off every once in a while, or the light won’t come in." [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/007d4a1214289aea09b9759ae1324e96.png)

🔹 [Memory-Constrained Policy Optimization](https://arxiv.org/pdf/2204.09315.pdf) 😶 

In addition to using the proximity of one single old policy as the first trust region as done by prior works, we propose to form a second trust region through the construction of another virtual policy that represents a wide range of past policies.

🔹 [A Temporal-Difference Approach to Policy Gradient Estimation](https://arxiv.org/pdf/2202.02396.pdf) :fire: 🌋 

TDRC: By using temporaldifference updates of the gradient critic from an off-policy data stream, we develop the first estimator that side-steps the distribution shift issue in a model-free way. [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/f516dfb84b9051ed85b89cdc3a8ab7f5.png) 

🔹 [gamma-models: Generative Temporal Difference Learning for Infinite-Horizon Prediction](https://arxiv.org/pdf/2010.14496.pdf) :fire: 🌋 :boom: 

Our goal is to make long-horizon predictions without the need to repeatedly apply a single-step model. 

🔹 [Generalised Policy Improvement with Geometric Policy Composition](https://proceedings.mlr.press/v162/thakoor22a/thakoor22a.pdf) 🌋 💧 

GGPI: 

🔹 [Taylor Expansions of Discount Factors](http://proceedings.mlr.press/v139/tang21b/tang21b.pdf) 🌋 💧 

We study the effect that this discrepancy of discount factors has during learning, and discover a family of objectives that interpolate value functions of two distinct discount factors. 

🔹 [Learning Retrospective Knowledge with Reverse Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/e6cbc650cd5798a05dfd0f51d14cde5c-Paper.pdf) 

Since such questions (how much fuel do we expect a car to have given it is at B at time t?) emphasize the influence of possible past events on the present, we refer to their answers as retrospective knowledge. We show how to represent retrospective knowledge with Reverse GVFs, which are trained via Reverse RL. [see GenTD] 

🔹 [A Generalized Bootstrap Target for Value-Learning, Efficiently Combining Value and Feature Predictions](https://www.aaai.org/AAAI22Papers/AAAI-12966.GX-ChenA.pdf) :fire: 

We focus on bootstrapping targets used when estimating value functions, and propose a new backup target, the lambda-return mixture, which implicitly combines value-predictive knowledge (used by TD methods) with (successor) feature-predictive knowledge—with a parameter lambda capturing how much to rely on each.

🔹 [A Deeper Look at Discounting Mismatch in Actor-Critic Algorithms](https://arxiv.org/pdf/2010.01069.pdf) :fire: 

We then propose to interpret the discounting in the critic in terms of a bias-variance-representation trade-off and provide supporting empirical results. In the second scenario, we consider optimizing a discounted objective (gamma < 1) and propose to interpret the omission of the discounting in the actor update from an auxiliary task perspective and provide supporting empirical results.

🔹 [An Analytical Update Rule for General Policy Optimization](https://proceedings.mlr.press/v162/li22d/li22d.pdf) :+1: :fire: :fire: 

The contributions of this paper include: (1) a new theoretical result that tightens existing bounds for local policy search using trust-region methods; (2) a closed-form update rule for general stochastic policies with monotonic improvement guarantee; [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/e1ab840a08f6e72d3baf13622bef60ad.png)

🔹 [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/pdf/2108.13264.pdf) :fire: 🌋

With the aim of increasing the field’s confidence in reported results with a handful of runs, we advocate for reporting interval estimates of aggregate performance and propose performance profiles to account for the variability in results, as well as present more robust and efficient aggregate metrics, such as interquartile mean scores, to achieve small uncertainty in results. [[rilable]](https://github.com/google-research/rliable)

🔹 [Safe Policy Improvement Approaches and their Limitations](https://arxiv.org/pdf/2208.00724.pdf) 

SPIBB

🔹 [BSAC: Bayesian Strategy Network Based Soft Actor-Critic in Deep Reinforcement Learning](https://arxiv.org/pdf/2208.06033.pdf) 😶 

(BSAC) model by organizing several sub-policies as a joint policy

<a name="anchor-marl"></a>

## MARL

- MARL <https://cloud.tencent.com/developer/article/1618396>

- A Survey on Transfer Learning for Multiagent Reinforcement Learning Systems :+1: :sweat_drops:

🔹 [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/pdf/1705.08926.pdf) :fire: 

COMA: to address the challenges of multi-agent credit assignment, it uses a counterfactual baseline that marginalises out a single agent’s action, while keeping the other agents’ actions fixed.

🔹 [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/pdf/1706.05296.pdf)  :+1:

VDN: aims to learn an optimal linear value decomposition from the team reward signal, by back-propagating the total Q gradient through deep neural networks representing the individual component value functions.

🔹 [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1803.11485.pdf) :fire: 🌋 

QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations.

- INTRINSIC REWARD

  🔹 [Hierarchical Cooperative Multi-Agent Reinforcement Learning with Skill Discovery](https://arxiv.org/pdf/1912.03558.pdf) :fire: :+1:  ​ ​

  The set of low-level skills emerges from an intrinsic reward that solely promotes the decodability of latent skill variables from the trajectory of a low-level skill, without the need for hand-crafted rewards for each skill.  

  🔹 [The Emergence of Individuality in Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2006.05842.pdf) :fire: :fire: :+1:

  EOI learns a probabilistic classifier that predicts a probability distribution over agents given their observation and gives each agent an intrinsic reward of being correctly predicted by the classifier.

  🔹 [A Maximum Mutual Information Framework for Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2006.02732.pdf) :+1: :droplet:  ​ ​
  
  introducing a latent variable to induce nonzero mutual information between actions.

  🔹 [MASER: Multi-Agent Reinforcement Learning with Subgoals Generated from Experience Replay Buffer](https://arxiv.org/pdf/2206.10607.pdf) :+1: 🔥 :fire: 

  MASER automatically generates proper subgoals for multiple agents from the experience replay buffer by considering both individual Q-value and total Qvalue. MASER designs individual intrinsic reward for each agent based on actionable representation relevant to Q-learning. 
  
- Learning Latent Representations

  🔹 [Learning Latent Representations to Influence Multi-Agent Interaction](https://arxiv.org/pdf/2011.06619.pdf) :+1:  ​

  We propose a reinforcement learningbased framework for learning latent representations of an agent’s policy, where the ego agent identifies the relationship between its behavior and the other agent’s future strategy. The ego agent then leverages these latent dynamics to influence the other agent, purposely guiding them towards policies suitable for co-adaptation.
  
- communicative partially-observable stochastic game (Comm-POSG)

-  

  🔹 [TRUST REGION POLICY OPTIMISATION IN MULTI-AGENT REINFORCEMENT LEARNING](https://arxiv.org/pdf/2109.11251.pdf) :volcano:

  We extend the theory of trust region learning to MARL. Central to our findings are the multi-agent advantage decomposition lemma and the sequential policy update scheme. Based on these, we develop Heterogeneous-Agent Trust Region Policy Optimisation (HATPRO) and Heterogeneous-Agent Proximal Policy Optimisation (HAPPO) algorithms.

<a name="anchor-constrainedrl"></a>  

## Constrained RL

🔹 [A Unified View of Entropy-Regularized Markov Decision Processes](https://arxiv.org/pdf/1705.07798.pdf) :fire: :droplet:  ​

using the conditional entropy of the joint state-action distributions as regularization yields a dual optimization problem closely resembling the Bellman optimality equations

🔹 [A Theory of Regularized Markov Decision Processes](https://arxiv.org/pdf/1901.11275.pdf) :+1::volcano: :boom: :droplet:

We have introduced a general theory of regularized MDPs, where the usual Bellman evaluation operator is modified by either a fixed convex function or a Bregman divergence between consecutive policies. We shown how many (variations of) existing algorithms could be derived from this general algorithmic scheme, and also analyzed and discussed the related propagation of errors.

🔹 [Mirror Learning: A Unifying Framework of Policy Optimisation](https://proceedings.mlr.press/v162/grudzien22a/grudzien22a.pdf) :+1: 😶 

we introduce a novel theoretical framework, named Mirror Learning, which provides theoretical guarantees to a large class of algorithms, including TRPO and PPO.

🔹 [Munchausen Reinforcement Learning](https://arxiv.org/pdf/2007.14430.pdf) :+1: ​ :fire:  :volcano: :droplet:

Yet, another estimate could be leveraged to bootstrap RL: the current policy. Our core contribution stands in a very simple idea: adding the scaled log-policy to the immediate reward.

🔹 [Leverage the Average: an Analysis of KL Regularization in Reinforcement Learning](https://arxiv.org/pdf/2003.14089.pdf) :+1: :fire: :volcano: :boom: :boom: :droplet:  ​

Convex Conjugacy for KL and Entropy Regularization;  1) Mirror Descent MPI: SAC, Soft Q-learning; Softmax DQN, mellowmax policy, TRPO, MPO, DPP, CVI:droplet:;  2) Dual Averaging MPI:droplet::  

🔹 [Proximal Iteration for Deep Reinforcement Learning](https://arxiv.org/pdf/2112.05848.pdf) :fire:

Our contribution is to employ Proximal Iteration for optimization in deep RL.  

🔹 [Theoretical Analysis of Efficiency and Robustness of Softmax and Gap-Increasing Operators in Reinforcement Learning](http://proceedings.mlr.press/v89/kozuno19a/kozuno19a.pdf) :+1: :droplet:

We propose and analyze conservative value iteration (CVI), which unifies value iteration, soft value iteration, advantage learning, and dynamic policy programming.

🔹 [Momentum in Reinforcement Learning](http://proceedings.mlr.press/v108/vieillard20a/vieillard20a.pdf) :+1: :fire:

We derive Momentum Value Iteration (MoVI), a variation of Value iteration that incorporates this momentum idea. Our analysis shows that this allows MoVI to average errors over successive iterations.

🔹 [Geometric Value Iteration: Dynamic Error-Aware KL Regularization for Reinforcement Learning](https://arxiv.org/pdf/2107.07659.pdf) :+1: :fire: :volcano:

we propose a novel algorithm, Geometric Value Iteration (GVI), that features a dynamic error-aware KL coefficient design with the aim of mitigating the impact of errors on performance. Our experiments demonstrate that GVI can effectively exploit the trade-off between learning speed and robustness over uniform averaging of a constant KL coefficient.

🔹 [Near Optimal Policy Optimization via REPS](https://openreview.net/pdf?id=ZEhDWKLTvt7) :volcano: :droplet:

Relative entropy policy search (REPS)

🔹 [ON COVARIATE SHIFT OF LATENT CONFOUNDERS IN IMITATION AND REINFORCEMENT LEARNING](https://openreview.net/pdf?id=w01vBAcewNX) :volcano: :droplet:

We consider the problem of using expert data with unobserved confounders for imitation and reinforcement learning.

🔹 [Constrained Policy Optimization](https://arxiv.org/pdf/1705.10528.pdf) :+1:  :fire:  :fire:  :volcano:  ​

We propose Constrained Policy Optimization (CPO), the first general-purpose policy search algorithm for constrained reinforcement learning with guarantees for near-constraint satisfaction at each iteration.

🔹 [Reward Constrained Policy Optimization](https://arxiv.org/pdf/1805.11074.pdf) :+1: :fire:  ​ ​

we present a novel multi-timescale approach for constrained policy optimization, called ‘Reward Constrained Policy Optimization’ (RCPO), which uses an alternative penalty signal to guide the policy towards a constraint satisfying one.

🔹 [PROJECTION-BASED CONSTRAINED POLICY OPTIMIZATION](https://arxiv.org/pdf/2010.03152.pdf) :+1: :fire:  ​ ​

the first step performs a local reward improvement update, while the second step reconciles any constraint violation by projecting the policy back onto the constraint set.

🔹 [First Order Constrained Optimization in Policy Space](https://proceedings.neurips.cc//paper/2020/file/af5d5ef24881f3c3049a7b9bfe74d58b-Paper.pdf) :fire: :+1:  

Using data generated from the current policy, FOCOPS first finds the optimal update policy by solving a constrained optimization problem in the nonparameterized policy space. FOCOPS then projects the update policy back into the parametric policy space.

🔹 [Reinforcement Learning with Convex Constraints](https://arxiv.org/pdf/1906.09323.pdf) :fire: :volcano: :droplet:

we propose an algorithmic scheme that can handle a wide class of constraints in RL tasks, specifically, any constraints that require expected values of some vector measurements (such as the use of an action) to lie in a convex set.  

🔹 [Batch Policy Learning under Constraints](https://arxiv.org/pdf/1903.08738.pdf) :volcano: :droplet:  ​ ​

propose a flexible meta-algorithm that admits any batch reinforcement learning and online learning procedure as subroutines.

🔹 [A Primal-Dual Approach to Constrained Markov Decision Processes](https://arxiv.org/pdf/2101.10895.pdf) :volcano: :droplet:  

 🔹 [Reward is enough for convex MDPs](https://arxiv.org/pdf/2106.00661.pdf) :fire: :volcano: :boom: :droplet:  ​

It is easy to see that Convex MDPs in which goals are expressed as convex functions of stationary distributions cannot, in general, be formulated in this manner (maximising a cumulative reward).  

🔹 [Challenging Common Assumptions in Convex Reinforcement Learning](https://arxiv.org/pdf/2202.01511.pdf) :+1: :fire:

We show that erroneously optimizing the infinite trials objective in place of the actual finite trials one, as it is usually done, can lead to a significant approximation error.

🔹 [DENSITY CONSTRAINED REINFORCEMENT LEARNING](https://openreview.net/pdf?id=jMc7DlflrMC) :+1:  :volcano:  ​

We prove the duality between the density function and Q function in CRL and use it to develop an effective primal-dual algorithm to solve density constrained reinforcement learning problems.  

🔹 [Control Regularization for Reduced Variance Reinforcement Learning](http://proceedings.mlr.press/v97/cheng19a/cheng19a.pdf) :fire: :volcano:

CORERL: we regularize the behavior of the deep policy to be similar to a policy prior, i.e., we regularize in function space. We show that functional reg. yields a bias-variance trade-off, and propose an adaptive tuning strategy to optimize this trade-off.

🔹 [REGULARIZATION MATTERS IN POLICY OPTIMIZATION - AN EMPIRICAL STUDY ON CONTINUOUS CONTROL](https://arxiv.org/pdf/1910.09191.pdf) :no_mouth:

We present the first comprehensive study of regularization techniques with multiple policy optimization algorithms on continuous control tasks.

🔹 [REINFORCEMENT LEARNING WITH SPARSE REWARDS USING GUIDANCE FROM OFFLINE DEMONSTRATION](https://arxiv.org/pdf/2202.04628.pdf) :fire: :volcano: :boom:

The proposed algorithm, which we call the Learning Online with Guidance Offline (LOGO) algorithm, merges a policy improvement step with an additional policy guidance step by using the offline demonstration data.

🔹 [MIRROR DESCENT POLICY OPTIMIZATION](https://openreview.net/pdf?id=aBO5SvgSt1)  :+1: :fire:

We derive on-policy and off-policy variants of MDPO (mirror descent policy optimization), while emphasizing important design choices motivated by the existing theory of MD in RL.

🔹 [BREGMAN GRADIENT POLICY OPTIMIZATION](https://openreview.net/pdf?id=ZU-zFnTum1N) :fire: :fire:

We propose a Bregman gradient policy optimization (BGPO) algorithm based on both the basic momentum technique and mirror descent iteration.

🔹 [Safe Policy Improvement by Minimizing Robust Baseline Regret](https://arxiv.org/pdf/1607.03842.pdf)  [see more in <a href="#anchor-offline">offline_rl</a>]

🔹 [Safe Policy Improvement with Baseline Bootstrapping](https://arxiv.org/pdf/1712.06924.pdf) :+1:  :fire: :volcano:

Our approach, called SPI with Baseline Bootstrapping (SPIBB), is inspired by the *knows-what-it-knows* paradigm: it bootstraps the trained policy with the baseline when the uncertainty is high.

🔹 [Safe Policy Improvement with Soft Baseline Bootstrapping](https://arxiv.org/pdf/1907.05079.pdf) :+1: :fire: :volcano:

Instead of binarily classifying the state-action pairs into two sets (the uncertain and the safe-to-train-on ones), we adopt a softer strategy that controls the error in the value estimates by constraining the policy change according to the local model uncertainty.

🔹 [SPIBB-DQN: Safe batch reinforcement learning with function approximation](https://www.microsoft.com/en-us/research/uploads/prod/2019/04/RLDM___SPIBB_DQN-2.pdf) 

🔹 [Safe policy improvement with estimated baseline bootstrapping](https://arxiv.org/pdf/1909.05236.pdf) 

🔹 [Incorporating Explicit Uncertainty Estimates into Deep Offline Reinforcement Learning](https://arxiv.org/pdf/2206.01085.pdf) :fire

deep-SPIBB: Evaluation step regularization + Uncertainty.

🔹 [Accelerating Safe Reinforcement Learning with Constraint-mismatched Baseline Policies](http://proceedings.mlr.press/v139/yang21i/yang21i.pdf) :fire: :volcano:

SPACE: We propose an iterative policy optimization algorithm that alternates between maximizing expected return on the task, minimizing distance to the baseline policy, and projecting the policy onto the constraint satisfying set.

🔹 [Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning](https://arxiv.org/pdf/2112.07701.pdf) :volcano: :fire:

We propose Conservative and Adaptive Penalty (CAP), a model-based safe RL framework that accounts for potential modeling errors by capturing model uncertainty and adaptively exploiting it to balance the reward and the cost objectives.

🔹 [Learning to be Safe: Deep RL with a Safety Critic](https://arxiv.org/pdf/2010.14603.pdf) :no_mouth:

We propose to learn how to be safe in one set of tasks and environments, and then use that learned intuition to constrain future behaviors when learning new, modified tasks.  

🔹 [CONSERVATIVE SAFETY CRITICS FOR EXPLORATION](https://arxiv.org/pdf/2010.14497.pdf) :fire: :volcano: :boom:

CSC:  we target the problem of safe exploration in RL by learning a conservative safety estimate of environment states through a critic, and provably upper bound the likelihood of catastrophic failures at every training iteration.

🔹 [Conservative Distributional Reinforcement Learning with Safety Constraints](https://arxiv.org/pdf/2201.07286.pdf) :fire:

We propose the CDMPO algorithm to solve safety-constrained RL problems. Our method incorporates a conservative exploration strategy as well as a conservative distribution function.  CSC + distributional RL + MPO + WAPID

🔹 [CUP: A Conservative Update Policy Algorithm for Safe Reinforcement Learning](https://arxiv.org/pdf/2202.07565.pdf) :+1: :fire: :volcano: :boom:

(i) We provide a rigorous theoretical analysis to extend the surrogate functions to generalized advantage estimator (GAE). GAE significantly reduces variance empirically while maintaining a tolerable level of bias, which is an efficient step for us to design CUP; (ii) The proposed bounds are tighter than existing works, i.e., using the proposed bounds as surrogate functions are better local approximations to the objective and safety constraints. (iii) The proposed CUP provides a non-convex implementation via first-order optimizers, which does not depend on any convex approximation.  

🔹 [Constrained Variational Policy Optimization for Safe Reinforcement Learning](https://proceedings.mlr.press/v162/liu22b/liu22b.pdf) :fire: 💧 

CVPO: [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/f4a331b7a22d1b237565d8813a34d8ac.png)

🔹 [A Review of Safe Reinforcement Learning: Methods,
Theory and Applications](https://arxiv.org/pdf/2205.10330.pdf) 💦

🔹 [MESA: Offline Meta-RL for Safe Adaptation and Fault Tolerance](https://arxiv.org/pdf/2112.03575.pdf) :no_mouth:

We cast safe exploration as an offline metaRL problem, where the objective is to leverage examples of safe and unsafe behavior across a range of environments to quickly adapt learned risk measures to a new environment with previously unseen dynamics.  

🔹 [Safe Driving via Expert Guided Policy Optimization](https://proceedings.mlr.press/v164/peng22a/peng22a.pdf) :+1: :fire:

We develop a novel EGPO method which integrates the guardian in the loop of reinforcement learning. The guardian is composed of an expert policy to generate demonstration and a switch function to decide when to intervene.  

🔹 [EFFICIENT LEARNING OF SAFE DRIVING POLICY VIA HUMAN-AI COPILOT OPTIMIZATION](https://arxiv.org/pdf/2202.10341.pdf) :fire:

 Human-AI Copilot Optimization (HACO): Human can take over the control and demonstrate to the agent how to avoid probably dangerous situations or trivial behaviors.

🔹 [SAFER: DATA-EFFICIENT AND SAFE REINFORCEMENT LEARNING THROUGH SKILL ACQUISITION](https://arxiv.org/pdf/2202.04849.pdf) :fire:

We propose SAFEty skill pRiors, a behavioral prior learning algorithm that accelerates policy learning on complex control tasks, under safety constraints. Through principled contrastive training on safe and unsafe data, SAFER learns to extract a safety variable from offline data that encodes safety requirements, as well as the safe primitive skills over abstract actions in different scenarios.

🔹 [Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees](https://arxiv.org/pdf/2201.08355.pdf) :fire:

We propose the Sim-to-Lab-to-Real framework that combines Hamilton-Jacobi reachability analysis and PAC-Bayes generalization guarantees to safely close the sim2real gap. Joint training of a performance and a backup policy in Sim training (1st stage) ensures safe exploration during Lab training (2nd stage).  

🔹 [Reachability Constrained Reinforcement Learning](https://arxiv.org/pdf/2205.07536.pdf) 🌋 

 this paper proposes the reachability CRL (RCRL) method by using reachability analysis to establish the novel self-consistency condition and characterize the feasible sets. The feasible sets are represented by the safety value function. 

Multi-Objective RL:

🔹 [Offline Constrained Multi-Objective Reinforcement Learning via Pessimistic Dual Value Iteration](https://proceedings.neurips.cc/paper/2021/file/d5c8e1ab6fc0bfeb5f29aafa999cdb29-Paper.pdf) :fire:

🔹 [Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer](https://proceedings.mlr.press/v162/alegre22a/alegre22a.pdf) :fire: :fire: 

We showed that any transfer learning problem within the SF framework can be mapped into an equivalent problem of learning multiple policies in MORL under linear preferences. We then introduced a novel SF-based extension of the OLS algorithm (SFOLS) to iteratively construct a set of policies whose SFs form a CCS. [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/da0dba87d95286d836e37ca60ab1e734_LY3UWtT.png) 

<a name="anchor-disrl"></a>

## Distributional RL

- Distributional RL [Hao Liang, CUHK](https://rlseminar.github.io/2019/03/11/hao.html)  [slide](https://rlseminar.github.io/static/files/RL_seminars2019-0311hao_distributional_final.pdf) :sweat_drops: :sweat_drops:  ​ ​

  🔹 C51: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf) :sweat_drops:  ​

  🔹 [CS598 - Statistical rl - NanJiang](https://www.bilibili.com/video/av929950486) :sweat_drops: ​

  🔹 [Information-Theoretic Considerations in Batch Reinforcement Learning](http://proceedings.mlr.press/v97/chen19e/chen19e.pdf) :volcano: :confused: :sweat_drops:

  🔹 [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/pdf/1806.06923.pdf) :sweat_drops:  ​
  
<a name="anchor-cl"></a>

## Continual Learning

  🔹 [Continual Learning with Deep Generative Replay](https://papers.nips.cc/paper/6892-continual-learning-with-deep-generative-replay.pdf)  :droplet: :no_mouth:  

We propose the Deep Generative Replay, a novel framework with a cooperative dual model architecture consisting of a deep generative model (“generator”) and a task solving model (“solver”).

  🔹 online learning; regret :sweat_drops:  ​

  🔹 [RESET-FREE LIFELONG LEARNING WITH SKILL-SPACE PLANNING](https://arxiv.org/pdf/2012.03548.pdf) :fire: 

  We propose Lifelong Skill Planning (LiSP), an algorithmic framework for non-episodic lifelong RL based on planning in an abstract space of higher-order skills. We learn the skills in an unsupervised manner using intrinsic rewards and plan over the learned skills using a learned dynamics model.

  🔹 [Don’t Start From Scratch: Leveraging Prior Data to Automate Robotic Reinforcement Learning](https://arxiv.org/pdf/2207.04703.pdf) :+1: 

  Our main contribution is demonstrating that incorporating prior data into a reinforcement learning system simultaneously addresses several key challenges in real-world robotic RL: sample-efficiency, zero-shot generalization, and autonomous non-episodic learning. 

  🔹 [A State-Distribution Matching Approach to Non-Episodic Reinforcement Learning](https://arxiv.org/pdf/2205.05212.pdf) :+1: :fire: 

  Assuming access to a few demonstrations, we propose a new method, MEDAL, that trains the backward policy to match the state distribution in the provided demonstrations. [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/41f860e3b7f548abc1f8b812059137bf.png)

  🔹 [You Only Live Once: Single-Life Reinforcement Learning via Learned Reward Shaping](https://openreview.net/pdf?id=weR4H5eEpv) :fire: 🌋 

  SLRL. (QWALE) that addresses the dearth of supervision by employing a distribution matching strategy that leverages the agent’s prior experience as guidance in novel situations.

<a name="anchor-selfpaced"></a>   <a name="anchor-curriculum"></a>

## Self-paced & Curriculum RL

  🔹 [Self-Paced Contextual Reinforcement Learning](https://arxiv.org/pdf/1910.02826.pdf) :volcano: :sweat_drops:  ​ ​

  We introduce a novel relative entropy reinforcement learning algorithm that gives the agent the freedom to control the intermediate task distribution, allowing for its gradual progression towards the target context distribution.  

  🔹 [Self-Paced Deep Reinforcement Learning](https://arxiv.org/pdf/2004.11812.pdf) :volcano: :sweat_drops:  ​ ​

  In this paper, we propose an answer by interpreting the curriculum generation as an inference problem, where distributions over tasks are progressively learned to approach the target task. This approach leads to an automatic curriculum generation, whose pace is controlled by the agent, with solid theoretical motivation and easily integrated with deep RL algorithms.

  🔹[Learning with AMIGO: Adversarially Motivated Intrinsic Goals](https://arxiv.org/pdf/2006.12122.pdf) :+1:   [Lil'Log-Curriculum](https://lilianweng.github.io/lil-log/2020/01/29/curriculum-for-reinforcement-learning.html) :+1:  ​

(Intrinsic motivation + Curriculum learning)

<a name="anchor-quadruped"></a>  

## Quadruped

- Locomotion

  🔹 [Reinforcement Learning with Evolutionary Trajectory Generator: A General Approach for Quadrupedal Locomotion](https://arxiv.org/pdf/2109.06409.pdf) :fire:

  ETG-RL: Unlike prior methods that use a fixed trajectory generator, the generator continually optimizes the shape of the output trajectory for the given task, providing diversified motion priors to guide the policy learning.

  🔹 [REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer](https://proceedings.mlr.press/v162/liu22p/liu22p.pdf) 😶 

  [[poster]](https://icml.cc/media/PosterPDFs/ICML%202022/33e8075e9970de0cfea955afd4644bb2_CuGkecl.png) 

- 

<a name="anchor-optimization"></a>  

## Optimization

- Contrastive Divergence (CD)

  🔹 [Training Products of Experts by Minimizing Contrastive Divergence](http://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf) :fire: :+1:    [Notes](https://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf) :+1:  ​

  C: contrastive = perceivable difference(s)

  D: divergence = general trend of such differences (over epochs)

  🔹 [A Contrastive Divergence for Combining Variational Inference and MCMC](https://arxiv.org/pdf/1905.04062.pdf) :droplet:  ​

  🔹 [CONTRASTIVE DIVERGENCE LEARNING IS A TIME REVERSAL ADVERSARIAL GAME](https://openreview.net/pdf?id=MLSvqIHRidA) :fire:  :droplet:

  Specifically, we show that CD is an adversarial learning procedure, where a discriminator attempts to classify whether a Markov chain generated from the model has been time-reversed.

- DISTRIBUTIONALLY ROBUST OPTIMIZATION (DRO)

  🔹 [MODELING THE SECOND PLAYER IN DISTRIBUTIONALLY ROBUST OPTIMIZATION](https://openreview.net/pdf?id=ZDnzZrTqU9N) :+1: :fire:  ​ ​

  we argue instead for the use of neural generative models to characterize the worst-case distribution, allowing for more flexible and problem-specific selection of *the uncertainty set*.

  🔹 [Wasserstein Distributionally Robust Optimization: Theory and Applications in Machine Learning](https://arxiv.org/pdf/1908.08729.pdf)

  🔹 [Variance-based regularization with convex objectives](https://arxiv.org/pdf/1610.02581.pdf)

  🔹 [Adaptive Regularization for Adversarial Training](https://arxiv.org/pdf/2206.03353.pdf) 🌋

  we develop a new data-adaptive regularization algorithm for adversarial training called Anti-Robust Weighted Regularization (ARoW). (more methods: PGD-Training, TRADES, GAIR-AT, FAT, MMA)

- Distribution shift; Robust;

  🔹 [Rethinking Importance Weighting for Deep Learning under Distribution Shift](https://arxiv.org/pdf/2006.04662.pdf) :confused:  ​

  🔹 [Variational Inference based on Robust Divergences](http://proceedings.mlr.press/v84/futami18a/futami18a.pdf) :+1: :droplet:

  Maximum Likelihood Estimation and Its Robust Variants. density power divergence; the β-divergence. γ-divergence;

  🔹 [A New Kind of Adversarial Example](https://arxiv.org/pdf/2208.02430.pdf) 👍 

   we consider the opposite which is adversarial examples that can fool a human but not a model

- Implicit learning

  🔹 [Generalization Bounded Implicit Learning of Nearly Discontinuous Functions](https://arxiv.org/pdf/2112.06881.pdf) :fire: :fire:

  🔹 [A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf) :volcano: :boom:

  Energy-Based Models (EBMs) capture dependencies between variables by associating a scalar energy to each configuration of the variables. Inference + Learning

  🔹 [Implicit Generation and Modeling with Energy-Based Models](https://papers.nips.cc/paper/2019/file/378a063b8fdb1db941e34f4bde584c7d-Paper.pdf) :+1:

  We present an algorithm and techniques for training energy based models that scale to challenging high-dimensional domains.

  🔹 [Compositional Visual Generation with Energy Based Models](https://arxiv.org/pdf/2004.06030.pdf) :+1:

  🔹 [Improved Contrastive Divergence Training of Energy-Based Model](https://arxiv.org/pdf/2012.01316.pdf) :fire: :volcano:

  We show that a gradient term neglected in the popular contrastive divergence formulation is both tractable to estimate and is important in avoiding training instabilities that previously limited applicability and scalability of energy-based models.

  🔹 [How to Train Your Energy-Based Models](https://arxiv.org/pdf/2101.03288.pdf) :fire: :fire:

  We start by explaining maximum likelihood training with Markov chain Monte Carlo (MCMC), and proceed to elaborate on MCMC-free approaches, including Score Matching (SM) and Noise Constrastive Estimation (NCE).  

  🔹 [A UNIFIED CONTRASTIVE ENERGY-BASED MODEL FOR UNDERSTANDING THE GENERATIVE ABILITY OF ADVERSARIAL TRAINING](https://arxiv.org/pdf/2203.13455.pdf) 🌋 

  CEM: the first probabilistic characterization of AT through a unified understanding of robustness and generative ability,  interprets unsupervised contrastive learning as animportant sampling of CEM.

  🔹 [YOUR CLASSIFIER IS SECRETLY AN ENERGY BASED MODEL AND YOU SHOULD TREAT IT LIKE ONE](https://arxiv.org/pdf/1912.03263.pdf) :fire: 🌋 

  We propose to reinterpret a standard discriminative classifier of p(y|x) as an energy based model for the joint distribution p(x, y). 

- Diffussion 

  🔹 [Sliced Score Matching: A Scalable Approach to Density and Score Estimation](http://proceedings.mlr.press/v115/song20a/song20a.pdf) :fire: 

   We show this difficulty (computing the Hessian of logdensity functions) can be mitigated by projecting the scores onto random vectors before comparing them.

  🔹 [Generative Modeling by Estimating Gradients of the Data Distribution](https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) :fire: 

  NCSN: we perturb the data with different levels of Gaussian noise, and jointly estimate the corresponding scores, i.e., the vector fields of gradients of the perturbed data distribution for all noise levels. For sampling, we propose an annealed Langevin dynamics where we use gradients corresponding to gradually decreasing noise levels as the sampling process gets closer to the data manifold.
  
  🔹 [Improved Techniques for Training Score-Based Generative Models](https://proceedings.neurips.cc/paper/2020/file/92c3b916311a5517d9290576e3ea37ad-Paper.pdf) :+1: 

  🔹 [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) :fire: 🌋 :boom: 

  Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. 

  🔹 [Improved Denoising Diffusion Probabilistic Models](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) :+1: 

  🔹 [SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/2011.13456.pdf) 🌋 

  Using SED, encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities.

  🔹 [Score-based Generative Modeling in Latent Space](https://proceedings.neurips.cc/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf) 

  🔹 [Back to the Source: Diffusion-Driven Test-Time Adaptation](https://arxiv.org/pdf/2207.03442.pdf) :+1: 🔥

  We instead update the target data, by projecting all test inputs toward the source domain with a generative diffusion model. Our diffusion-driven adaptation method, DDA, shares its models for classification and generation across all domains.

  🔹 [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/pdf/2205.09991.pdf) :fire: 

  🔹 [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/pdf/2208.06193.pdf) :fire: 

  we propose Diffusion-QL that utilizes a conditional diffusion model as a highly expressive policy class for behavior cloning and policy regularization.

- Data Valuation 

  🔹 [Data shapley: Equitable valuation of data for machine learning](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf) 



  🔹 [DATA VALUATION USING REINFORCEMENT LEARNING](https://arxiv.org/pdf/1909.11671.pdf) :fire: 

  DVRL:  We train the data value estimator using a reinforcement signal of the reward obtained on a small validation set that reflects performance on the  target task. 

  🔹 [Data Valuation for Offline Reinforcement Learning](https://arxiv.org/pdf/2205.09550.pdf) :fire: 

  DVORL: allows us to identify relevant and high-quality transitions, improving the performance and transferability of policies learned by offline reinforcement learning algorithms.

- IMOP, IOP: Inverse (Multiobjective) Optimization Problem 
  
  🔹 [Expert Learning through Generalized Inverse Multiobjective Optimization: Models, Insights, and Algorithms](http://proceedings.mlr.press/v119/dong20f/dong20f.pdf)



- others

  🔹 [Structured Prediction with Partial Labelling through the Infimum Loss](http://proceedings.mlr.press/v119/cabannnes20a/cabannnes20a.pdf) :+1: :droplet:  ​ ​
  
  🔹 [Bridging the Gap Between f-GANs and Wasserstein GANs](https://arxiv.org/pdf/1910.09779.pdf) :+1: :fire:  :volcano: ​​
  
  we propose an new training objective where we additionally optimize over a set of importance weights over the generated samples. By suitably constraining the feasible set of importance weights, we obtain a family of objectives which includes and generalizes the original f-GAN and WGAN objectives.
  
  🔹 [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://papers.nips.cc/paper/2016/file/cedebb6e872f539bef8c3f919874e9d7-Paper.pdf) :+1: :fire:  
  
  🔹 [Discriminator Contrastive Divergence: Semi-Amortized Generative Modeling by Exploring Energy of the Discriminator​](https://arxiv.org/pdf/2004.01704.pdf) 👍 :fire: 

  DCD: Compared to standard GANs, where the generator is directly utilized to obtain new samples, our method proposes a semi-amortized generation procedure where the samples are produced with the generator’s output as an initial state. 

  🔹 [DISCRIMINATOR REJECTION SAMPLING](https://arxiv.org/pdf/1810.06758.pdf) :fire: 

  We ask if the information retained in the weights of the discriminator at the end of the training procedure can be used to “improve” the generator.
  
  🔹 [On Symmetric Losses for Learning from Corrupted Labels](https://arxiv.org/pdf/1901.09314.pdf) :fire: :droplet:
  
   using a symmetric loss is advantageous in the balanced error rate (BER) minimization and area under the receiver operating characteristic curve (AUC) maximization from corrupted labels.
  
  🔹 [A Symmetric Loss Perspective of Reliable Machine Learning](https://arxiv.org/pdf/2101.01366.pdf) :+1: :droplet: :fire: ​
  
  a symmetric loss can yield robust classification from corrupted labels in balanced error rate (BER) minimization and area under the receiver operating characteristic curve (AUC) maximization.
  
  🔹 [Connecting Generative Adversarial Networks and Actor-Critic Methods](https://arxiv.org/pdf/1610.01945.pdf) :+1:  ​
  
  GANs can be seen as a modified actor-critic method with blind actors (stateless) in stateless MDPs.
  
  🔹 [An Image is Worth More Than a Thousand Words: Towards Disentanglement in the Wild](https://arxiv.org/pdf/2106.15610.pdf) :+1: :volcano:  
  
  we propose a method for disentangling a set of factors which are only partially labeled, as well as separating the complementary set of residual factors that are never explicitly specified.  ​
  
  🔹 [Recomposing the Reinforcement Learning Building Blocks with Hypernetworks](https://arxiv.org/pdf/2106.06842.pdf) :fire:
  
  To consider the interaction between the input variables, we suggest using a Hypernetwork architecture where a primary network determines the weights of a conditional dynamic network.  
  
  🔹 [SUBJECTIVE LEARNING FOR OPEN-ENDED DATA](https://arxiv.org/pdf/2108.12113.pdf) :fire:
  
  OSL: we present a novel supervised learning framework of learning from open-ended data, which is modeled as data implicitly sampled from multiple domains with the data in each domain obeying a domain-specific target function.

  🔹 [The State of Sparse Training in Deep Reinforcement Learning](https://arxiv.org/pdf/2206.10369.pdf) 

  The State of Sparse Training in Deep Reinforcement Learning

  🔹 [Learning Iterative Reasoning through Energy Minimization](https://arxiv.org/pdf/2206.15448.pdf) :fire: 

  We train a neural network to parameterize an energy landscape over all outputs, and implement each step of the iterative reasoning as an energy minimization step to find a minimal energy solution.

  🔹 [Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting](https://proceedings.neurips.cc/paper/2019/file/d76d8deea9c19cc9aaf2237d2bf2f785-Paper.pdf) :fire: 🌋 

  A standard technique to correct this bias is importance sampling, where samples from the model are weighted by the likelihood ratio under model and true distributions. When the likelihood ratio is unknown, it can be estimated by training a probabilistic classifier to distinguish samples from the two distributions. 

  🔹 [Telescoping Density-Ratio Estimation](https://proceedings.neurips.cc/paper/2020/file/33d3b157ddc0896addfb22fa2a519097-Paper.pdf) :+1: 

  we introduce a new framework, telescoping density-ratio estimation (TRE), that enables the estimation of ratios between highly dissimilar densities in high-dimensional spaces.

+ Distillation 
  
  🔹 [Policy Distillation with Selective Input Gradient Regularization for Efficient Interpretability](https://arxiv.org/pdf/2205.08685.pdf) 🔥

  We propose an approach of Distillation with selective Input Gradient Regularization (DIGR) which uses policy distillation and input gradient regularization to produce new policies that achieve both high interpretability and computation efficiency in generating saliency maps. 

  🔹 [Gradient-based Bi-level Optimization for Deep Learning: A Survey](https://arxiv.org/pdf/2207.11719.pdf) :+1: :fire: 🌋 :boom: :boom: 

   Bi-level optimization embeds one problem within another and the gradient-based category solves the outer level task by computing the hypergradient. 

  
## Galaxy  Forest  

​ :milky_way: :snowflake: :cyclone: :ocean: :volcano: :earth_africa: :earth_americas: :earth_asia: :book: :dart: :gem: :lemon: :headphones: :pushpin: :artificial_satellite: :satellite: :rocket: :stars: :sunrise_over_mountains: :triangular_flag_on_post: :beer: :tea: :date: :golf: :hourglass: :camera: :pager: :balloon: :trophy: :apple: :rice: ​

- Deep Reinforcement Learning amidst Lifelong Non-Stationarity <https://arxiv.org/pdf/2006.10701.pdf>

- Learning Robot Skills with Temporal Variational Inference <https://arxiv.org/pdf/2006.16232.pdf>

- Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design <https://arxiv.org/pdf/0912.3995.pdf> [icml2020 test of time award] :confused: :question:

- On Learning Sets of Symmetric Elements <https://arxiv.org/pdf/2002.08599.pdf> [icml2020 outstanding paper awards] :confused: :question:

- Non-delusional Q-learning and Value Iteration <https://papers.nips.cc/paper/8200-non-delusional-q-learning-and-value-iteration.pdf> [NeurIPS2018 Best Paper Award]

- SurVAE **Flows**: Surjections to Bridge the Gap between VAEs and Flows [Max Welling] <https://arxiv.org/pdf/2007.02731.pdf>

  🔹 [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/pdf/1908.09257.pdf)  :+1: ; Citing:  [Normalizing Flows for Probabilistic Modeling and Inference](https://arxiv.org/pdf/1912.02762.pdf) :+1: :boom: :boom: :boom: ; [lil-log: Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html) ; [Jianlin Su: f-VAES](https://zhuanlan.zhihu.com/p/45090025) :sweat_drops: ; [Deep generative models](https://deepgenerativemodels.github.io/notes/) :sweat_drops:  [Another slide](https://drive.google.com/file/d/1SGGWQR_FCHzsg-_aYzCjgHypXNesl6ML/view);  

  🔹 [Deep Kernel Density Estimation](https://zhuanlan.zhihu.com/p/73426787) (Maximum Likelihood, Neural Density Estimation (Auto Regressive Models + Normalizing Flows), Score Matching ([MRF](http://signal.ee.psu.edu/mrf.pdf)), Kernel Exponential Family ([RKHS](http://songcy.net/posts/story-of-basis-and-kernel-part-2/)), Deep Kernel);  

  🔹 [Machine Theory of Mind](https://arxiv.org/pdf/1802.07740.pdf) 

  ToM 

- Self-Supervised Learning  [lil-log](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html) :sweat_drops: ;

  🔹 [Self-Supervised Exploration via Disagreement](https://arxiv.org/pdf/1906.04161.pdf)  :confused:

  🔹
  
- 🔹 [Comparing Distributions by Measuring Differences that Affect Decision Making](https://openreview.net/pdf?id=KB5onONJIAU) :+1: :fire: :volcano: :boom:  

  H divergence: (H Entropy) We propose a new class of discrepancies based on the optimal loss for a decision task – two distributions are different if the optimal decision loss is higher on their mixture than on each individual distribution. By suitably choosing the decision task, this generalizes the JS divergence and the MMD family.

  🔹 [RotoGrad: Gradient Homogenization in Multitask Learning](https://arxiv.org/pdf/2103.02631.pdf) :+1: :volcano: :boom:

  We introduce RotoGrad, an algorithm that tackles negative transfer as a whole: it jointly homogenizes gradient magnitudes and directions, while ensuring training convergence. The proposed strategy is to introduce additional parameterized rotation matrices, each of which modifies the shared representation before it is passed to a corresponding task-specific branch. The parameters of these rotation matrices are optimized to maximize gradient similarity between different tasks at the branch point; this optimization step is interlaced with standard updates of other network parameters to minimize total task loss.

  🔹 [META DISCOVERY: LEARNING TO DISCOVER NOVEL CLASSES GIVEN VERY LIMITED DATA](https://openreview.net/pdf?id=MEpKGLsY8f) :+1: :fire: :fire:
  
  [Demystifying](https://arxiv.org/pdf/2102.04002.pdf) Assumptions in Learning to Discover Novel Classes (L2DNC): find that high-level semantic features should be shared among the seen and unseen classes. CATA ( Clustering-rule-aware Task Sampler): Data have multiple views. However, there are always one view or a few views that are dominate for each instance, and these dominated views are similar with high-level semantic meaning. We propose to use dominated views to replace with clustering rules.
  
  🔹 [Learning Surrogate Losses](https://arxiv.org/pdf/1905.10108.pdf) :fire:
  
  We learn smooth relaxation versions of the true losses by approximating them through a surrogate neural network.
  
  🔹 [Learning Surrogates via Deep Embedding](https://arxiv.org/pdf/2007.00799.pdf) :fire:
  
  Training neural networks by minimizing learned surrogates that approximate the target evaluation metric.
  
  🔹 [RELATIONAL SURROGATE LOSS LEARNING](https://openreview.net/pdf?id=dZPgfwaTaXv) :fire: :volcano:
  
  Instead of directly approximating the evaluation metrics as previous methods, this paper proposes a new learning method by revisiting the purpose of loss functions, which is to distinguish the performance of models. Hence, the authors aim to learn the surrogate losses by making the surrogate losses have the same discriminability as the evaluation metrics. The idea is straightforward and is easy to implement by using ranking correlation as an optimization objective.
  
  🔹 [Iterative Teacher-Aware Learning](https://openreview.net/pdf?id=aLkuboH1SQX) :+1: :fire: :fire:
  
  We propose a gradient optimization based teacher-aware learner who can incorporate teacher’s cooperative intention into the likelihood function and learn provably faster compared with the naive learning algorithms used in previous machine teaching works.
  
  🔹 [MAXIMIZING ENSEMBLE DIVERSITY IN DEEP REINFORCEMENT LEARNING](https://openreview.net/pdf?id=hjd-kcpDpf2) :+1:
  
  We describe Maximize Ensemble Diversity in Reinforcement Learning (MED-RL), a set of regularization methods inspired from the economics and consensus optimization to improve diversity in the ensemble based deep reinforcement learning methods by encouraging inequality between the networks during training.
  
- AAA:  
  
  🔹 [Efficiently Identifying Task Groupings for Multi-Task Learning](https://arxiv.org/pdf/2109.04617.pdf) :fire:
  
  Our method determines task groupings in a single run by training all tasks together and quantifying the effect to which one task’s gradient would affect another task’s loss.  
  
  🔹 [Learning from Failure: Training Debiased Classifier from Biased Classifier](https://arxiv.org/pdf/2007.02561.pdf) :+1: :volcano:
  
  Our idea is twofold; (a) we intentionally train the first network to be biased by repeatedly amplifying its “prejudice”, and (b) we debias the training of the second network by focusing on samples that go against the prejudice of the biased network in (a).
  
  🔹 [Just Train Twice: Improving Group Robustness without Training Group Information](http://proceedings.mlr.press/v139/liu21f/liu21f.pdf) :+1:
  
  We propose a simple two-stage approach, JTT, that minimizes the loss over a reweighted dataset (second stage) where we upweight training examples that are misclassified at the end of a few steps of standard training (first stage).
  
   🔹 [Environment Inference for Invariant Learning](http://proceedings.mlr.press/v139/creager21a/creager21a.pdf)
  
  🔹 [Robustness between the worst and average case](https://openreview.net/pdf?id=Y8YqrYeFftd) :fire: :droplet:
  
  We proposed a definition of intermediate-q robustness that smooths the gap between robustness to random perturbations and adversarial robustness by generalizing these notions of robustness as functional `q norms of the loss function over the perturbation distribution.
  
  🔹 [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://openreview.net/pdf?id=-zgb2v8vV_w) :+1: :fire:
  
  We aim to learn models that adapt at test time to domain shift using unlabeled test points. Our primary contribution is to introduce the framework of adaptive risk minimization (ARM), in which models are directly optimized for effective adaptation to shift by learning to adapt on the training domains.
  
  🔹 [Unsupervised Learning of Compositional Energy Concepts](https://openreview.net/pdf?id=2RgFZHCrI0l) :fire: :fire:
  
  We propose COMET, which discovers and represents concepts as separate energy functions, enabling us to represent both global concepts as well as objects under a unified framework. COMET discovers energy functions through recomposing the input image, which we find captures independent factors without additional supervision.

  🔹 [LEARNING A LATENT SEARCH SPACE FOR ROUTING PROBLEMS USING VARIATIONAL AUTOENCODERS](https://openreview.net/pdf?id=90JprVrJBO) :+1: :fire: 

  CVAE-Opt: This paper proposes a learning based approach for solving combinatorial optimization problems such as routine using continuous optimizers. The key idea is to learn a continuous latent space via conditional VAE to represent solutions and perform search in this latent space for new problems at the test-time.

  🔹 [Learning to Solve Vehicle Routing Problems: A Survey](https://arxiv.org/pdf/2205.02453.pdf) :+1: 

  🔹 [Time Series Forecasting Models Copy the Past: How to Mitigate](https://arxiv.org/pdf/2207.13441.pdf) :+1: :fire: 🌋 

  In the presence of noise and uncertainty, neural network models tend to replicate the last observed value of the time series, thus limiting their applicability to real-world data. We also propose a regularization term penalizing the replication of previously seen values.

  🔹 [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://arxiv.org/pdf/2106.04399.pdf) :fire: 🌋

  GFlowNet: based on a view of the generative process as a flow network, making it possible to handle the tricky case where different trajectories can yield the same final state, e.g., there are many ways to sequentially add atoms to generate some molecular graph.

  🔹 [Biological Sequence Design with GFlowNets](https://arxiv.org/pdf/2203.04115.pdf) :fire: 

  We also propose a scheme to incorporate existing labeled datasets of candidates, in addition to a reward function, to speed up learning in GFlowNets.

- Interpretation 
  
  🔹 [CONTRASTIVE EXPLANATIONS FOR REINFORCEMENT LEARNING VIA EMBEDDED SELF PREDICTIONS](https://openreview.net/pdf?id=Ud3DSz72nYR) :+1: 

  We investigate a deep reinforcement learning (RL) architecture that supports explaining why a learned agent prefers one action over another.
  
- Label Noise
  
  🔹 [Eliciting Informative Feedback: The Peer-Prediction Method](https://presnick.people.si.umich.edu/papers/elicit/FinalPrePub.pdf) :fire: :volcano: :boom:
  
  Each rater merely reports a signal, and the system applies proper scoring rules to the implied posterior beliefs about another rater's report.
  
  🔹 [Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates](https://arxiv.org/pdf/1910.03231.pdf) :volcano:
  
  We introduce a new family of loss functions that we name as peer loss functions, which enables learning from noisy labels and does not require a priori specification of the noise rates.  
  
  🔹 [LEARNING WITH INSTANCE-DEPENDENT LABEL NOISE: A SAMPLE SIEVE APPROACH](https://arxiv.org/pdf/2010.02347.pdf) :volcano: :+1:
  
  CORES2: We propose to train a classifier using a novel confidence regularization (CR) term and theoretically guarantee that, under mild assumptions, minimizing the confidence regularized cross-entropy (CE) loss on the instance-based noisy distribution is equivalent to minimizing the pure CE loss on the corresponding “unobservable” clean distribution.
  
  🔹 [A Second-Order Approach to Learning with Instance-Dependent Label Noise](https://arxiv.org/pdf/2012.11854.pdf) :fire: :droplet:
  
  We propose and study the potentials of a second-order approach that leverages the estimation of several covariance terms defined between the instance-dependent noise rates and the Bayes optimal label. We show that this set of second-order statistics successfully captures the induced imbalances.
  
  🔹 [Does label smoothing mitigate label noise?](https://arxiv.org/pdf/2003.02819.pdf) :fire:
  
  We related smoothing to one of these correction techniques, and re-interpreted it as a form of regularisation.
  
  🔹 [Understanding Generalized Label Smoothing when Learning with Noisy Labels](https://arxiv.org/pdf/2106.04149.pdf) :volcano:
  
  We unify label smoothing with either positive or negative smooth rate into a generalized label smoothing (GLS) framework. We proceed to show that there exists a phase transition behavior when finding the optimal label smoothing rate for GLS.
  
  🔹 [Understanding Instance-Level Label Noise: Disparate Impacts and Treatments](https://arxiv.org/pdf/2102.05336.pdf) :droplet: :fire:
  
  🔹 [WHEN OPTIMIZING f -DIVERGENCE IS ROBUST WITH LABEL NOISE](https://arxiv.org/pdf/2011.03687.pdf) :volcano:
  
  We derive a nice decoupling property for a family of f-divergence measures when label noise presents, where the divergence is shown to be a linear combination of the variational difference defined on the clean distribution and a bias term introduced due to the noise.
  
  🔹 [Can Less be More? When Increasing-to-Balancing Label Noise Rates Considered Beneficial](https://arxiv.org/pdf/2107.05913.pdf)
  
  We are primarily inspired by three observations: 1) In contrast to reducing label noise rates, increasing the noise rates is easy to implement; 2) Increasing a certain class of instances’ label noise to balance the noise rates (increasing-to-balancing) results in an easier learning problem; 3) Increasing-to-balancing improves fairness guarantees against label bias.
  
- Semi-supervise; self-training
  
  🔹 [TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING](https://openreview.net/pdf?id=BJ6oOfqge) :fire:
  
  We introduce self-ensembling, where we form a consensus prediction of the unknown labels using the outputs of the network-in-training on different epochs, and most importantly, under different regularization and input augmentation conditions.
  
  🔹 [Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data](https://arxiv.org/pdf/2010.03622.pdf) :volcano: :sweat_drops:
  
  This work provides a unified theoretical analysis of selftraining with deep networks for semi-supervised learning, unsupervised domain adaptation, and unsupervised learning. At the core of our analysis is a simple but realistic “expansion” assumption, which states that a low-probability subset of the data must expand to a neighborhood with large probability relative to the subset. We also assume that neighborhoods of examples in different classes have minimal overlap.
  
  🔹 [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf) :fire:
  
  The teacher produces high-quality pseudo labels by reading in clean images, while the student is required to reproduce those labels with augmented images as input.
  
  🔹 [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/pdf/1904.12848.pdf) :+1:
  
  We present a new perspective on how to effectively noise unlabeled examples and argue that the quality of noising, specifically those produced by advanced data augmentation methods, plays a crucial role in semi-supervised learning.
  
  🔹 [Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning](https://arxiv.org/pdf/1704.03976.pdf) :volcano:
  
  Virtual adversarial loss is defined as the robustness of the conditional label distribution around each input data point against local perturbation. Unlike adversarial training, our method defines the adversarial direction without label information and is hence applicable to semi-supervised learning.
  
  🔹 [Semi-supervised Learning by Entropy Minimization](https://proceedings.neurips.cc/paper/2004/file/96f2b50b5d3613adf9c27049b2a888c7-Paper.pdf) :fire:
  
  🔹 [Robustness to Adversarial Perturbations in Learning from Incomplete Data](https://arxiv.org/pdf/1905.13021.pdf) :volcano: :droplet:
  
  We unify two major learning frameworks: Semi-Supervised Learning (SSL) and Distributionally Robust Learning (DRL).
  
  🔹 [SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2021/papers/Prabhu_SENTRY_Selective_Entropy_Optimization_via_Committee_Consistency_for_Unsupervised_Domain_ICCV_2021_paper.pdf) :volcano:
  
  A UDA algorithm that judges the reliability of a target instance based on its predictive consistency under a committee of random image transformations.
  
  🔹 [Deep Co-Training with Task Decomposition for Semi-Supervised Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Deep_Co-Training_With_Task_Decomposition_for_Semi-Supervised_Domain_Adaptation_ICCV_2021_paper.pdf)

  🔹 [Debiased Contrastive Learning](https://proceedings.neurips.cc/paper/2020/file/63c3ddcc7b23daa1e42dc41f9a44a873-Paper.pdf) 🌋 

  we develop a debiased contrastive objective that corrects for the sampling of same-label datapoints, even without knowledge of the true labels.

  🔹 [Positive Unlabeled Contrastive Learning](https://arxiv.org/pdf/2206.01206.pdf) :fire: 

  puNCE: that leverages the available explicit (from labeled samples) and implicit (from unlabeled samples) supervision to learn useful representations from positive unlabeled input data.

  🔹 [Boosting Few-Shot Learning With Adaptive Margin Loss](https://arxiv.org/pdf/2005.13826.pdf) :fire: 🌋 

  This paper proposes an adaptive margin principle to improve the generalization ability of metric-based meta-learning approaches for few-shot learning problems.

- Uncertainty (Calibration)

  🔹 [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) 🔥 

  🔹 [Accurate Uncertainties for Deep Learning Using Calibrated Regression](https://arxiv.org/pdf/1807.00263.pdf) :fire:

  🔹 [Calibrated Reliable Regression using Maximum Mean Discrepancy](https://arxiv.org/pdf/2006.10255.pdf) :+1:👍

  We propose the calibrated regression method using the maximum mean discrepancy by minimizing the kernel embedding measure.


  
  
- GAN

   🔹 [A Unified View of cGANs with and without Classifiers](https://openreview.net/pdf?id=j6KoGtzPYa) :fire: :volcano:

  This submission proposes to analyze the most popular variations of conditional GANs (ACGAN, ProjGAN, ContraGAN) under a unified, energy-based, formulation (ECGAN).

<a name="anchor-pareto"></a>

- Pareto

  🔹 [Pareto Multi-Task Learning](https://arxiv.org/pdf/1912.12854.pdf) :boom: :boom: :fire: :droplet:

  we proposed a novel Pareto Multi-Task Learning (Pareto MTL) algorithm to generate a set of well-distributed Pareto solutions with different trade-offs among tasks for a given multi-task learning (MTL) problem.

  🔹 [Efficient Continuous Pareto Exploration in Multi-Task Learning](https://arxiv.org/pdf/2006.16434.pdf) [zhihu](https://zhuanlan.zhihu.com/p/159000150) :boom: :fire: :+1:  ​ ​ ​

  🔹 [Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control](https://proceedings.icml.cc/static/paper_files/icml/2020/1114-Paper.pdf) :sweat_drops:  ​
  
  🔹 [Pareto Domain Adaptation](https://openreview.net/pdf?id=frgb7FsKWs3) :volcano:
  
  To reach a desirable solution on the target domain, we design a surrogate loss mimicking target classification. To improve target-prediction accuracy to support the mimicking, we propose a target-prediction refining mechanism which exploits domain labels via Bayes’ theorem.

  🔹 [PARETO POLICY POOL FOR MODEL-BASED OFFLINE REINFORCEMENT LEARNING](https://openreview.net/pdf/811177d23b2117fa0be0cc22952e7c1e3325bf59.pdf) 
  
  🔹 [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) :+1: :fire:
  
   CAGrad: minimizes the average loss function, while leveraging the worst local improvement of individual tasks to regularize the algorithm trajectory. CAGrad balances the objectives automatically and still provably converges to a minimum over the average loss.

   🔹 [Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization](https://arxiv.org/pdf/2206.04496.pdf) :fire:

  We show how to detect the sub-graphs in the computational graphs where gradients conflict (impartiality blocks), as well as how to leverage existing gradient-conflict solutions from multitask learning to mitigate modality collapse.
  
- BNN

  🔹 [Auto-Encoding Variational Bayes](https://www.ics.uci.edu/~welling/publications/papers/AEVB_ICLR14.pdf) :+1:  ​

  🔹 [Unlabelled Data Improves Bayesian Uncertainty Calibration under Covariate Shift](http://proceedings.mlr.press/v119/chan20a/chan20a.pdf)  :+1: :fire:

  we develop an approximate Bayesian inference scheme based on posterior regularisation, wherein unlabelled target data are used as “pseudo-labels” of model confidence that are used to regularise the model’s loss on labelled source data.

  🔹 [Understanding Uncertainty in Bayesian Deep Learning](https://arxiv.org/pdf/2106.13055.pdf) :sweat_drops:

- HyperNetworks

  🔹 [HYPERNETWORKS](https://arxiv.org/pdf/1609.09106.pdf) :+1:  

  🔹 [NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING](https://arxiv.org/pdf/1611.01578.pdf)

  🔹 [META-LEARNING WITH LATENT EMBEDDING OPTIMIZATION](https://arxiv.org/pdf/1807.05960.pdf) :fire: :volcano:

  LEO: learning a data-dependent latent generative representation of model parameters, and performing gradient-based meta-learning in this low dimensional latent space.

  🔹 [CONTINUAL LEARNING WITH HYPERNETWORKS](https://arxiv.org/pdf/1906.00695.pdf) :fire:

  Instead of recalling the input-output relations of all previously seen data, task-conditioned hypernetworks only require rehearsing task-specific weight realizations, which can be maintained in memory using a simple regularizer.
  
  🔹 [Continual Model-Based Reinforcement Learning with Hypernetworks](https://meta-learn.github.io/2020/papers/52_paper.pdf) :+1:
  
  🔹 [Hypernetwork-Based Augmentation](https://arxiv.org/pdf/2006.06320.pdf) :fire:
  
  We propose an efficient gradient-based search algorithm, called Hypernetwork-Based Augmentation (HBA), which simultaneously learns model parameters and augmentation hyperparameters in a single training.

  🔹 [Goal-Conditioned Generators of Deep Policies](https://arxiv.org/pdf/2207.01570.pdf) :fire: 

  Using context commands of the form “generate a policy that achieves a desired expected return,” our NN generators combine powerful exploration of parameter space with generalization across commands to iteratively find better and better policies.

  🔹 [General Policy Evaluation and Improvement by Learning to Identify Few But Crucial States](https://arxiv.org/pdf/2207.01566.pdf) :fire: 

  Here we combine the actor-critic architecture of ParameterBased Value Functions and the policy embedding of Policy Evaluation Networks to learn a single value function for evaluating (and thus helping to improve) any policy represented by a deep neural network (NN).

  🔹 [Policy Evaluation Networks](https://arxiv.org/pdf/2002.11833.pdf) :fire: :+1: 

  PVN: We introduced a network that can generalize in policy space, by taking policy fingerprints as inputs. These fingerprints are differentiable policy embeddings obtained by inspecting the policy’s behaviour in a set of key states.

  🔹 [PARAMETER-BASED VALUE FUNCTIONS](https://arxiv.org/pdf/2006.09226.pdf) :fire: 

  We introduce a class of value functions called Parameter-Based Value Functions (PBVFs) whose inputs include the policy parameters. They can generalize across different policies.
  
## Aha

### Alpha

- Gaussian Precess, Kernel Method, [EM](https://zhuanlan.zhihu.com/p/54823479), [Conditional Neural Process](https://zhuanlan.zhihu.com/p/142260457), [Neural Process](https://zhuanlan.zhihu.com/p/70226367),  (Deep Mind, ICML2018) :+1: ​
- [Weak Duality](https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture7.pdf), [Fenchel-Legendre Duality](https://zhuanlan.zhihu.com/p/34236792), [Convex-Optimization](https://glooow1024.github.io/categories/Convex-Optimization/), [Convex-Optimization - bilibili](https://www.bilibili.com/video/BV1Jt411p7jE?p=20&spm_id_from=pageDriver),  
- [Online Convex Optimization](https://zhuanlan.zhihu.com/p/346763047), [ONLINE LEARNING](https://parameterfree.com/lecture-notes-on-online-learning/), [Convex Optimization (book)](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf),
- [Total variation distance](http://people.csail.mit.edu/jayadev/teaching/ece6980f16/scribing/26-aug-16.pdf),
- [Ising model](https://zhuanlan.zhihu.com/p/163706800), Gibbs distribution, [VAEBM](https://openreview.net/pdf?id=5m3SEczOV8L),
- [f-GAN](https://kexue.fm/archives/5977), [GAN-OP](https://zhuanlan.zhihu.com/p/50488424), [ODE: GAN](https://zhuanlan.zhihu.com/p/65953336),
- [Wasserstein Distance](Wasserstein Distance), [Statistical Aspects of Wasserstein Distances](https://arxiv.org/pdf/1806.05500.pdf), [Optimal Transport and Wasserstein Distance](http://www.stat.cmu.edu/~larry/=sml/Opt.pdf), [Metrics for GAN](An empirical study on evaluation metrics of generative adversarial networks), [Metrics for GAN zhihu](https://zhuanlan.zhihu.com/p/99375611), [MMD: Maximum Mean Discrepancy](https://www.zhihu.com/question/288185961),
- [MARKOV-LIPSCHITZ DEEP LEARNING](https://arxiv.org/pdf/2006.08256.pdf),
- [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) :sweat_drops: ,  ​
- [VC dimensition](https://zhuanlan.zhihu.com/p/41109051),
- [BALD](https://arxiv.org/pdf/1112.5745.pdf),

---

---

### Blog & Corp. & Legend

[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/), [OpenAI Blog](https://openai.com/blog/), [OpenAI Baselines](https://github.com/openai/baselines), [DeepMind](https://deepmind.com/blog?category=research), [BAIR](https://bair.berkeley.edu/blog/), [Stanford AI Lab](https://ai.stanford.edu/blog/),

[Lil'Log](https://lilianweng.github.io/lil-log/), [Andrej Karpathy blog](http://karpathy.github.io/), [The Gradient](https://thegradient.pub/), [RAIL - course - RL](http://rail.eecs.berkeley.edu/deeprlcourse-fa19/), [RAIL -  cs285](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc), [inFERENCe](https://www.inference.vc/),

[covariant](http://covariant.ai/),  [RL_theory_book](https://rltheorybook.github.io/),

UCB: [Tuomas Haarnoja](https://scholar.google.com/citations?hl=en&user=VT7peyEAAAAJ),  [Pieter Abbeel](https://scholar.google.com/citations?user=vtwH6GkAAAAJ&hl=en&oi=sra), [Sergey Levine](https://scholar.google.com/citations?user=8R35rCwAAAAJ&hl=en),  [Abhishek Gupta](https://people.eecs.berkeley.edu/~abhigupta/), [Coline Devin](https://people.eecs.berkeley.edu/~coline/), [YuXuan (Andrew) Liu](https://yuxuanliu.com/), [Rein Houthooft](https://scholar.google.com/citations?user=HBztuGIAAAAJ&hl=en), [Glen Berseth](https://scholar.google.com/citations?hl=en&user=-WZcuuwAAAAJ&view_op=list_works&sortby=pubdate),

UCSD: [Xiaolong Wang](https://scholar.google.com.sg/citations?hl=en&user=Y8O9N_0AAAAJ&sortby=pubdate&view_op=list_works&citft=1&email_for_op=liujinxin%40westlake.edu.cn&gmla=AJsN-F79NyYa7yONVJDI6gta02XaqE24ZGaLjgMRYiKHJ-wf2Sb4Y-ZqHrtaAGQPSWWphueUEd9d5l47a06Z1sWq91OSww8miQ),

CMU: [Benjamin Eysenbach](https://scholar.google.com/citations?hl=en&user=DRnOvU8AAAAJ&view_op=list_works&sortby=pubdate), [Ruslan Salakhutdinov](https://scholar.google.com/citations?hl=en&user=ITZ1e7MAAAAJ&view_op=list_works&sortby=pubdate),

Standord: [Chelsea Finn](https://scholar.google.com/citations?user=vfPE6hgAAAAJ&hl=en), [Tengyu Ma], [Tianhe Yu], [Rui Shu],

NYU: [Rob Fergus](https://scholar.google.com/citations?user=GgQ9GEkAAAAJ&hl=en&oi=sra),

MIT: [Bhairav Mehta](https://scholar.google.com/citations?hl=en&user=uPtOmHcAAAAJ),  [Leslie Kaelbling](https://scholar.google.com/citations?hl=en&user=IcasIiwAAAAJ&view_op=list_works&sortby=pubdate), [Joseph J. Lim](https://scholar.google.com/citations?hl=zh-CN&user=jTnQTBoAAAAJ&view_op=list_works&sortby=pubdate),

Caltech: [Joseph Marino](https://joelouismarino.github.io/), [Yisong Yue](https://scholar.google.com/citations?hl=en&user=tEk4qo8AAAAJ&view_op=list_works&sortby=pubdate) [Homepage](http://www.yisongyue.com/about.php),

DeepMind: [David Silver](https://scholar.google.com/citations?user=-8DNE4UAAAAJ&hl=en), [Yee Whye Teh](https://scholar.google.com/citations?user=y-nUzMwAAAAJ&hl=en) [[Homepage]](https://www.stats.ox.ac.uk/~teh/), [Alexandre Galashov](https://scholar.google.com/citations?user=kIpoNtcAAAAJ&hl=en&oi=sra), [Leonard Hasenclever](https://leonard-hasenclever.github.io/) [[GS]](https://scholar.google.com/citations?user=dD-3S4QAAAAJ&hl=en&oi=sra), [Siddhant M. Jayakumar](https://scholar.google.com/citations?user=rJUAY8QAAAAJ&hl=en&oi=sra), [Zhongwen Xu](https://scholar.google.com/citations?hl=en&user=T4xuHn8AAAAJ&view_op=list_works&sortby=pubdate), [Markus Wulfmeier](https://scholar.google.de/citations?hl=en&user=YCO3WQsAAAAJ&view_op=list_works&sortby=pubdate) [[HomePage]](https://markusrw.github.io/), [Wojciech Zaremba](https://scholar.google.com/citations?hl=en&user=XCZpOcAAAAAJ&view_op=list_works&sortby=pubdate), [Aviral Kumar](https://scholar.google.ca/citations?hl=en&user=zBUwaGkAAAAJ&sortby=pubdate&view_op=list_works&citft=1&email_for_op=liujinxin%40westlake.edu.cn&gmla=AJsN-F5E6ErpTneaGZx0cFbO3J7y-pEg7TpQXU2LTHXXoDUX79vsrOzPiGg55PiqYlz0GVVk5kJT8orQHvsGwW5WF7RSg-9ryv5Xo-L0rpHKJOuBiuL-dfE),

Google: [Ian Fischer](https://scholar.google.com/citations?hl=en&user=Z63Zf_0AAAAJ&view_op=list_works&sortby=pubdate), [Danijar Hafner](https://scholar.google.de/citations?hl=en&user=VINmGpYAAAAJ&view_op=list_works&sortby=pubdate) [[Homepage]](https://danijar.com/), [Ofir Nachum](https://scholar.google.com/citations?hl=en&user=C-ZlBWMAAAAJ&sortby=pubdate&view_op=list_works&citft=1&citft=3&email_for_op=liujinxin%40westlake.edu.cn&gmla=AJsN-F6bB1Pjv8yoTSFnbtB3GJE8dXTxX4wK1GnOvBUvWhOt8ZBNxojCh223i5_AvQ347yNG-MLSVENT3s-8UCe4DIDgvLNNG8kvQNxMjH7_VCrX6-P0FVQ), [Yinlam Chow](https://scholar.google.com/citations?hl=en&user=BFlpS-8AAAAJ&sortby=pubdate&view_op=list_works&citft=1&email_for_op=liujinxin%40westlake.edu.cn&gmla=AJsN-F7Ht8XHXMlvXRq2vMXdWh8tAT298ToP-ONtMyacd1uEqMWgiWBqT9SRimdl-c-xcemDY324kkrnfR9nfDbaq8sso-KZ0A), [Shixiang Shane Gu](https://scholar.google.com/citations?hl=en&user=B8wslVsAAAAJ&view_op=list_works&sortby=pubdate), [Mohammad Ghavamzadeh]

Montreal: [Anirudh Goyal](https://scholar.google.com/citations?hl=en&user=krrh6OUAAAAJ&view_op=list_works&sortby=pubdate) [Homepage](https://anirudh9119.github.io/),

Toronto: [Jimmy Ba](https://scholar.google.com/citations?hl=en&user=ymzxRhAAAAAJ&view_op=list_works&sortby=pubdate); [Amir-massoud Farahmand](https://scholar.google.com/citations?hl=zh-CN&user=G5SAV7gAAAAJ&view_op=list_works&sortby=pubdate);

Columbia: [Yunhao (Robin) Tang](https://robintyh1.github.io/),

OpenAI:

THU:  [Chongjie Zhang](https://scholar.google.com/citations?user=LjxqXycAAAAJ&hl=en) [[Homepage]](http://people.iiis.tsinghua.edu.cn/~zhang/), [Yi Wu](https://scholar.google.com/citations?hl=en&user=dusV5HMAAAAJ&view_op=list_works&sortby=pubdate), [Mingsheng Long](https://scholar.google.com/citations?user=_MjXpXkAAAAJ) [[Homepage]](http://ise.thss.tsinghua.edu.cn/~mlong/),

PKU: [Zongqing Lu](https://scholar.google.com/citations?user=k3IFtTYAAAAJ&hl=en),

NJU: [Yang Yu](https://scholar.google.com/citations?user=PG2lDSwAAAAJ&hl=en),

TJU: [Jianye Hao](https://scholar.google.com/citations?user=FCJVUYgAAAAJ&hl=en),

HIT: [PR&IS research center](http://pr-ai.hit.edu.cn/main.htm),

Salesforce : [Alexander Trott](https://scholar.google.com/citations?hl=en&user=rB4bvV0AAAAJ&view_op=list_works&sortby=pubdate),

Flowers Lab (INRIA): [Cédric Colas](https://scholar.google.fr/citations?hl=fr&user=VBz8gZ4AAAAJ&view_op=list_works&sortby=pubdate),

[NeurIPS](https://proceedings.neurips.cc/paper/),
