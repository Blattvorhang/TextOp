# TextOp: Real-time Interactive Text-Driven Humanoid Robot Motion Generation and Control

![Cover](docs/Cover.jpg)

[[**website**](https://text-op.github.io/)] | [[**demo**](https://youtu.be/nKxE7ff1FwY)] 


## News
- **[2026-02]** Release the official version of TextOp. Note: The latest code and dataset are not yet updated.
- \[2026-01\] Updated `TextOpDeploy` modules to fix several bugs and add support for keyboard-based joystick in `unitree_mujoco`. You can now run sim2sim experiments in `TextOpDeploy` more smoothly.  
- \[2025-11\] We release the preview version of TextOp, including **code**, **pretrained models** and **demo**.


## About
We propose TextOp, a novel framework for real-time, interactive, text-driven humanoid robot motion generation and control. It allows users to instruct the robot using natural language and modify commands on the fly, producing smooth, whole-body motions instantly.

Our system utilizes a two-layer architecture for execution. At the high level, a robot motion diffusion autoregressive model processes current user text commands to generate the kinematic motion trajectory. The low level employs a universal motion tracking policy for motor control. In this way, TextOp achieves both instant responsiveness and precise robot control.

TextOp is highly versatile and supports a wide range of behaviours, from simple gestures to complex motion sequences, all without pre-recorded scripts or manual programming. This approach provides a significantly more intuitive human-robot interaction paradigm, unlocking the potential for highly adaptable and easily controllable robots in real-world applications.

Key features:
- **End-to-end open-source pipeline** covering dataset construction, model training, and real-robot deployment.
- **High-fidelity motion tracking**: our universal Tracker policy achieves nearly 100% success per sequence on cleaned training data.
- **Clean and modular codebase**, designed for readability, maintainability, and easy extension.

## Repository Structure
```
TextOp/
│
├── TextOpRobotMDAR/        # High-level text-to-motion model
├── TextOpTracker/          # Low-level whole-body universal motion tracking policy
├── TextOpDeploy/           # Sim2sim and Sim2real deployment
├── dataset/                # Scripts for dataset processing
├── deps/                   # Third-party packages
└── docs/
```

We also provide the retargeted public datasets used in our experiments, as well as pretrained models for both RobotMDAR and Tracker policy. These resources enable you to reproduce our results out of the box.
> Our models are trained on a mixture of public datasets and a small private dataset. However, comparable performance should be achievable using only the public data.

## Usage

See [USAGE.md](USAGE.md) for details.


<!-- ## Citation

If you find our work helpful, please cite: -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.


## Acknowledgements
TextOpTracker is built upon [Beyondmimic](https://beyondmimic.github.io/). TextOpRobotMDAR is based on a reconstruction of [DART](https://github.com/zkf1997/DART) and is adapted for robot configurations.

We use publicly available human motion datasets, including [AMASS](https://amass.is.tue.mpg.de/) with [BABEL](https://babel.is.tue.mpg.de/)-[TEACH](https://download.is.tue.mpg.de/download.php?domain=teach&resume=1&sfile=babel-data/babel-teach.zip) annotations and [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), and employ [GMR](https://github.com/YanjieZe/GMR) for retargeting.

## Contact

Feel free to open an issue or discussion if you encounter any problems or have questions about this project.

For collaborations, feedback, or further inquiries, please reach out to:

- Weiji Xie: [xieweiji249@sjtu.edu.cn](mailto:xieweiji249@sjtu.edu.cn) or Weixin `shisoul`
- Jiakun Zheng: [zjk9098@gmail.com](mailto:zjk9098@gmail.com)
- Chenjia Bai: [baicj@chinatelecom.cn](mailto:baicj@chinatelecom.cn)
- You can also join our weixin discussion group for timely Q&A. Since the group already exceeds 200 members, you'll need to first add one of the authors on Weixin to receive an invitation to join.


We welcome contributions and are happy to support the community in building upon this work!