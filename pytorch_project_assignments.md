# PyTorch Learning Path — Project-Based Assignments

A hands-on, project-based curriculum to master PyTorch from the ground up. Each assignment builds on the previous one, so complete them in order.

---

## Assignment 1: Tensor Playground

**Goal:** Build intuition for PyTorch tensors and their operations.

**Tasks:**

1. **Image as a Tensor:** Load any image using PIL, convert it to a PyTorch tensor, and perform the following manipulations — flip it horizontally, crop the center 50%, convert it to grayscale by averaging the RGB channels, and adjust brightness by scaling pixel values. Save each result back as an image.

2. **Matrix Operations Calculator:** Write a Python script that takes two matrices as input (from the user or hardcoded) and performs addition, element-wise multiplication, matrix multiplication, transpose, determinant, and inverse. Use only PyTorch tensor operations — no NumPy allowed.

3. **Broadcasting Challenge:** Create a tensor of shape `(5, 1)` and another of shape `(1, 4)`. Predict the output shape of their addition, then verify. Repeat with shapes `(3, 1, 4)` and `(1, 2, 1)`. Write a short explanation of the broadcasting rules you observed.

**Deliverables:** A Jupyter notebook with code, outputs, and markdown cells explaining what each operation does.

---

## Assignment 2: Autograd — Build a Linear Regression From Scratch

**Goal:** Understand automatic differentiation and computational graphs.

**Tasks:**

1. Generate synthetic data: `y = 3x + 7 + noise` with 200 data points.
2. Initialize weight `w` and bias `b` as tensors with `requires_grad=True`.
3. Implement the training loop **without using `nn.Module` or any optimizer**:
   - Compute predictions: `y_pred = w * x + b`
   - Compute MSE loss manually
   - Call `.backward()` to compute gradients
   - Update `w` and `b` manually using gradient descent: `w = w - lr * w.grad`
   - Zero out gradients after each step
4. Plot the loss curve over 1000 epochs.
5. Print the learned `w` and `b` — they should be close to 3 and 7.
6. **Bonus:** Add a print statement inside the loop showing `w.grad` and `b.grad` for the first 5 epochs. Explain in a comment what these values represent.

**Deliverables:** Script or notebook with the training loop, loss plot, and final parameter values.

---

## Assignment 3: Your First Neural Network with `nn.Module`

**Goal:** Learn to define and use neural network modules.

**Tasks:**

1. **Build a Multi-Layer Perceptron (MLP)** that classifies the `make_moons` dataset from `sklearn.datasets`.
   - Create a class `MoonClassifier(nn.Module)` with at least 2 hidden layers
   - Use `ReLU` activations between layers
   - Use `Sigmoid` on the output for binary classification
2. Print the model architecture using `print(model)`.
3. Print the total number of trainable parameters.
4. Train the model and plot the **decision boundary** at epoch 0, 50, 200, and 500 by evaluating the model on a mesh grid.
5. Experiment: What happens when you use only 1 hidden layer with 2 neurons vs. 2 hidden layers with 32 neurons each? Show both decision boundaries side by side.

**Deliverables:** Notebook with model definition, training code, decision boundary visualizations, and a paragraph comparing the two architectures.

---

## Assignment 4: Loss Functions and Optimizers — Optimizer Showdown

**Goal:** Understand how different loss functions and optimizers behave.

**Tasks:**

1. Use the **Iris dataset** (from sklearn) to build a 3-class classifier.
2. Train the same model architecture **4 separate times**, each with a different optimizer: `SGD`, `SGD with Momentum`, `Adam`, and `RMSprop`. Keep learning rate constant at 0.01.
3. Record and plot the training loss and validation accuracy for each optimizer on the same graph.
4. **Learning Rate Experiment:** Pick the best optimizer from above. Train 3 more times with learning rates `0.001`, `0.01`, and `0.1`. Plot the loss curves together.
5. **Loss Function Swap:** Replace `CrossEntropyLoss` with `NLLLoss` (adding `LogSoftmax` to your model). Verify that results are equivalent.

**Deliverables:** Notebook with comparison plots and a written summary (5–10 sentences) of which optimizer worked best and why.

---

## Assignment 5: Custom Dataset and DataLoader — Build a CSV Classifier

**Goal:** Master data loading pipelines.

**Tasks:**

1. Download any tabular classification dataset (e.g., Titanic, Heart Disease, or Wine Quality from Kaggle/UCI).
2. Create a **custom `Dataset` class** that:
   - Reads the CSV in `__init__`
   - Handles missing values and encodes categorical features
   - Returns `(features_tensor, label_tensor)` in `__getitem__`
   - Returns the dataset length in `__len__`
3. Split data into train/val/test sets.
4. Create `DataLoader` instances with `batch_size=32`, `shuffle=True` for training, and `shuffle=False` for validation/test.
5. Write a training loop that iterates over the DataLoader and prints batch shapes for the first 3 batches to verify everything works.
6. Train a simple MLP on this data and report final test accuracy.
7. **Bonus:** Add a custom `Transform` class that normalizes features to zero mean and unit variance.

**Deliverables:** Notebook with the custom Dataset class, DataLoader usage, and trained model with test accuracy.

---

## Assignment 6: The Complete Training Pipeline — MNIST Digit Classifier

**Goal:** Tie together everything from Assignments 1–5 into a polished pipeline.

**Tasks:**

1. Load the MNIST dataset using `torchvision.datasets.MNIST` with transforms (convert to tensor, normalize).
2. Build a feedforward neural network (not CNN yet) with at least 2 hidden layers.
3. Implement a **complete training pipeline** with the following structure:
   - A `train_one_epoch(model, loader, optimizer, criterion)` function
   - A `validate(model, loader, criterion)` function
   - Proper use of `model.train()` and `model.eval()`
   - `torch.no_grad()` during validation
   - Tracking of training loss, validation loss, and validation accuracy per epoch
4. Train for 20 epochs and plot training loss vs. validation loss.
5. After training, display a grid of 16 test images with their **predicted labels** and **true labels**. Highlight incorrect predictions in red.
6. Print a **confusion matrix** for the test set.

**Deliverables:** A well-organized script or notebook with functions, plots, and the confusion matrix.

---

## Assignment 7: Model Serialization — Checkpoint and Resume Training

**Goal:** Learn to save, load, and resume models.

**Tasks:**

1. Using your MNIST model from Assignment 6:
   - Save the model's `state_dict` after epoch 10 to a `.pth` file
   - Save a **full checkpoint** (model state, optimizer state, epoch number, loss) to a separate file
2. Write a `load_checkpoint()` function that restores everything and resumes training from epoch 11.
3. Demonstrate that the model resumes correctly by:
   - Training for 10 epochs → save checkpoint → stop
   - Load checkpoint → train for 10 more epochs
   - Compare the final accuracy to a model trained for 20 epochs straight
4. **Bonus:** Save two models — one with `torch.save(model.state_dict())` and one with `torch.save(model)`. Load both and verify they produce identical outputs on the same input. Write a comment explaining which method is preferred and why.

**Deliverables:** Script with save/load functions and a comparison showing that resumed training matches continuous training.

---

## Assignment 8: GPU Training — Speed Benchmarking

**Goal:** Understand device management and GPU acceleration.

**Tasks:**

1. Modify your MNIST pipeline to support both CPU and GPU training via a single `device` variable.
2. Ensure all tensors and the model are moved to `device` using `.to(device)`.
3. **Benchmark:** Time training for 5 epochs on CPU vs. GPU. Print the time per epoch for each.
4. Build a **larger model** (e.g., 4 hidden layers with 512 neurons each) and repeat the benchmark. At what model size does GPU training start to significantly outperform CPU?
5. **Common Bug Challenge:** Intentionally create the error where model is on GPU but data is on CPU. Observe the error message, then fix it. Write a comment explaining the error.

**Deliverables:** Notebook with benchmarking results (table or bar chart) and a written analysis of when GPU training is worthwhile.

---

## Assignment 9: CNN — Image Classification on CIFAR-10

**Goal:** Build and train convolutional neural networks.

**Tasks:**

1. Load CIFAR-10 with data augmentation transforms: random horizontal flip, random crop with padding, normalization.
2. Build a CNN with the following architecture:
   - 3 convolutional blocks, each containing: `Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d`
   - A fully connected classifier head
3. Train for 30 epochs with Adam optimizer and a learning rate scheduler (`StepLR` or `CosineAnnealingLR`).
4. Plot training/validation accuracy and loss curves.
5. **Visualize learned filters:** Extract and display the filters from the first convolutional layer as a grid of small images.
6. **Per-class accuracy:** After training, compute and display accuracy for each of the 10 classes. Which classes does the model confuse most?
7. **Architecture Experiment:** Add or remove a convolutional block and compare performance. Does deeper always mean better?

**Deliverables:** Notebook with CNN code, training curves, filter visualizations, per-class accuracy report, and architecture comparison.

---

## Assignment 10: RNN/LSTM — Name Nationality Classifier

**Goal:** Learn sequence modeling with recurrent networks.

**Tasks:**

1. Use a names dataset (e.g., names from different countries/languages — available in PyTorch tutorials or on Kaggle).
2. **Data preprocessing:**
   - Convert each name (string) into a sequence of character indices
   - Pad sequences to equal length using `torch.nn.utils.rnn.pad_sequence`
   - Create a vocabulary mapping (char → index)
3. Build three models and compare them:
   - A vanilla `nn.RNN`
   - An `nn.LSTM`
   - An `nn.GRU`
4. Each model should read the character sequence and output a nationality prediction.
5. Train each for 20 epochs and compare their accuracy and training curves.
6. **Bonus — Name Generator:** Take your best model, flip it into a generative mode: given a nationality, generate new names character by character using the model's predictions. (Hint: use the hidden state from training.)

**Deliverables:** Notebook with data preprocessing, three model implementations, comparison chart, and (optional) generated names.

---

## Assignment 11: Transfer Learning — Cat vs. Dog Classifier

**Goal:** Fine-tune pretrained models for a new task.

**Tasks:**

1. Download a small subset (1000 images per class) of the Cats vs. Dogs dataset.
2. Load a pretrained `ResNet18` from `torchvision.models`.
3. **Strategy A — Feature Extraction:**
   - Freeze all layers except the final fully connected layer
   - Replace the FC layer with one that outputs 2 classes
   - Train for 10 epochs
4. **Strategy B — Fine-Tuning:**
   - Unfreeze the last residual block + FC layer
   - Use a smaller learning rate (e.g., 1e-4) for pretrained layers and a larger one (e.g., 1e-3) for the new FC layer using **parameter groups** in the optimizer
   - Train for 10 epochs
5. **Strategy C — From Scratch:**
   - Use the same ResNet18 architecture but with random weights (`pretrained=False`)
   - Train for 10 epochs
6. Compare all three strategies: accuracy, training time, and loss curves.
7. **Grad-CAM Visualization:** For your best model, generate Grad-CAM heatmaps on 5 test images to show what regions the model focuses on.

**Deliverables:** Notebook with three strategies compared, a summary table, and Grad-CAM visualizations.

---

## Assignment 12: Regularization Lab

**Goal:** Experiment with regularization techniques and understand their effects.

**Tasks:**

1. Use CIFAR-10 and your CNN from Assignment 9 as the baseline.
2. **Create 5 variants of the model**, each adding one regularization technique:
   - Variant A: Baseline (no regularization)
   - Variant B: Add `nn.Dropout(0.5)` after each ReLU
   - Variant C: Add weight decay (`1e-4`) to the optimizer
   - Variant D: Add `BatchNorm2d` after each convolution
   - Variant E: Add aggressive data augmentation (random rotation, color jitter, random erasing)
3. Train each variant for 30 epochs and track **both training and validation accuracy**.
4. Plot all 5 training curves and all 5 validation curves on two separate graphs.
5. Identify which variant has the **largest gap between training and validation accuracy** (most overfitting) and which has the **smallest gap**.
6. **Combined Model:** Create a final model that combines the best regularization techniques and train it. Does combining them help?

**Deliverables:** Notebook with 5+ model variants, comparison plots, and a written analysis identifying the most effective regularization strategy.

---

## Assignment 13: TensorBoard Dashboard

**Goal:** Learn to monitor training with TensorBoard.

**Tasks:**

1. Choose any model from a previous assignment and add TensorBoard logging.
2. Log the following to TensorBoard:
   - Scalar: training loss, validation loss, and accuracy per epoch
   - Histogram: weight distributions and gradient distributions for each layer (every 5 epochs)
   - Images: a batch of input images with predictions (every 10 epochs)
   - Model Graph: log the model architecture
   - Embedding: use `add_embedding()` to visualize the penultimate layer activations of 500 test samples, colored by class
3. Run TensorBoard and take screenshots of each visualization.
4. **Hyperparameter Comparison:** Run 3 experiments with different hyperparameters (e.g., different learning rates). Use `SummaryWriter` with different log directories so all runs appear on the same TensorBoard dashboard.
5. Write observations: What do the weight histograms tell you? Can you spot vanishing or exploding gradients?

**Deliverables:** Notebook with logging code, TensorBoard screenshots, and written observations.

---

## Assignment 14: Distributed Training — Multi-GPU CIFAR Trainer

**Goal:** Scale training across multiple GPUs (or simulate the setup).

**Tasks:**

1. Take your CIFAR-10 CNN and wrap it with `nn.DataParallel`. Train for 10 epochs. Compare throughput (images/second) to single-GPU training.
2. Refactor the code to use `DistributedDataParallel` (DDP):
   - Set up process groups
   - Use `DistributedSampler` for the DataLoader
   - Launch with `torch.multiprocessing.spawn`
3. If you only have 1 GPU, simulate multi-process training by spawning 2 processes that share the same GPU. Verify it runs without errors.
4. **Gradient Accumulation:** Implement gradient accumulation to simulate a batch size of 256 while only using a physical batch size of 32. Compare final accuracy to actual batch size 256.

**Deliverables:** Script with both DataParallel and DDP implementations, throughput comparison, and gradient accumulation demo.

---

## Assignment 15: Custom Autograd Function — Differentiable Image Filter

**Goal:** Extend PyTorch's autograd with custom forward and backward passes.

**Tasks:**

1. Implement a **custom Gaussian blur** as a `torch.autograd.Function`:
   - `forward`: Apply a Gaussian kernel to the input image tensor via convolution
   - `backward`: Manually compute the gradient of the loss with respect to the input
2. Verify your custom backward pass using `torch.autograd.gradcheck` with double precision tensors.
3. Build a small network: `Input Image → Your Custom Gaussian Blur → Conv2d → ReLU → Output`. Train it on a simple task (e.g., edge detection or denoising) and verify gradients flow through your custom function.
4. **Bonus:** Implement a custom `Swish` activation function (`x * sigmoid(x)`) with a hand-written backward pass. Compare results to the built-in `nn.SiLU()`.

**Deliverables:** Notebook with the custom autograd function, gradcheck verification, and a working training loop that uses it.

---

## Assignment 16: Model Export — TorchScript and ONNX

**Goal:** Optimize and export models for production deployment.

**Tasks:**

1. Take your best CIFAR-10 model and export it two ways:
   - **TorchScript Tracing:** `torch.jit.trace(model, sample_input)` — save and reload
   - **TorchScript Scripting:** `torch.jit.script(model)` — save and reload
2. Run inference with both the original and TorchScript models on 100 test images. Verify outputs match (use `torch.allclose`).
3. Benchmark inference speed: Original model vs. TorchScript model (average over 1000 forward passes).
4. **ONNX Export:**
   - Export the model to ONNX format using `torch.onnx.export`
   - Load it with `onnxruntime` and run inference
   - Compare the outputs to the original PyTorch model
5. **Edge Case:** Add an `if` statement in your model's `forward` method (e.g., `if x.shape[0] > 1`). Try tracing vs. scripting and explain why tracing fails.

**Deliverables:** Script with both export methods, speed benchmarks, ONNX inference demo, and a written explanation of tracing vs. scripting differences.

---

## Assignment 17: Mixed Precision Training

**Goal:** Speed up training with automatic mixed precision.

**Tasks:**

1. Take your CIFAR-10 CNN and implement mixed precision training using `torch.cuda.amp`:
   - Wrap the forward pass in `torch.autocast('cuda')`
   - Use `GradScaler` for gradient scaling
2. Train for 30 epochs with and without mixed precision. Compare:
   - Training time per epoch
   - GPU memory usage (use `torch.cuda.max_memory_allocated()`)
   - Final accuracy (should be nearly identical)
3. **Inspect dtypes:** During a forward pass with autocast, print the dtype of intermediate tensors (after conv layers, after batch norm, etc.) to see which operations use `float16` vs. `float32`.
4. **Stress Test:** Increase the model size until you run out of GPU memory without mixed precision, then show that mixed precision lets you train the larger model.

**Deliverables:** Notebook with side-by-side comparisons (time, memory, accuracy), dtype inspection output, and stress test results.

---

## Assignment 18: Transformer From Scratch — Tiny Shakespeare Text Generator

**Goal:** Understand attention mechanisms by building a transformer.

**Tasks:**

1. Download the Tiny Shakespeare dataset (or any small text corpus).
2. Implement a **character-level transformer** from scratch:
   - **Positional Encoding:** Implement sinusoidal positional encoding
   - **Self-Attention:** Implement scaled dot-product attention manually (no `nn.MultiheadAttention`)
   - **Multi-Head Attention:** Split into multiple heads, apply attention, concatenate
   - **Transformer Block:** Multi-head attention → Add & Norm → Feed-forward → Add & Norm
   - **Full Model:** Token embedding + positional encoding → N transformer blocks → linear output
3. Train on sequences of length 64, predicting the next character.
4. Generate text by sampling from the model autoregressively. Show 5 generated samples of 500 characters each at different temperatures (0.5, 0.8, 1.0, 1.2, 1.5).
5. **Attention Visualization:** For a given input sequence, extract and visualize the attention weights as a heatmap. Do different heads attend to different patterns?

**Deliverables:** Notebook with the full transformer implementation (no library shortcuts for attention), generated text samples, and attention heatmaps.

---

## Assignment 19: Hooks — Layer-by-Layer Network Introspection Tool

**Goal:** Use hooks for debugging and feature extraction.

**Tasks:**

1. Load a pretrained ResNet50 and register **forward hooks** on every layer to capture intermediate activations.
2. Pass a single image through the network and:
   - Print the shape of activations at every layer
   - Visualize the activations of the first 3 convolutional layers as image grids
   - Compute and display the mean activation magnitude per layer as a bar chart
3. Register **backward hooks** on every layer to capture gradients during backpropagation.
   - Pass an image, compute loss for a target class, and call `.backward()`
   - Plot the gradient magnitude per layer — can you spot if gradients vanish or explode?
4. **Feature Extraction Tool:** Build a reusable `FeatureExtractor` class that:
   - Takes any model and a list of layer names
   - Registers hooks on those layers
   - Returns a dictionary of `{layer_name: activation_tensor}` after a forward pass
   - Properly removes hooks when done (using hook handles)
5. Use your `FeatureExtractor` to compare activations of a cat image vs. a dog image at 3 different layers. What differences do you see?

**Deliverables:** Notebook with the FeatureExtractor class, activation visualizations, gradient plots, and cat-vs-dog comparison.

---

## Capstone Project: End-to-End Image Classification System

**Goal:** Combine everything into a production-quality project.

**Tasks:**

Build a complete image classification system for a dataset of your choice (e.g., Food-101, Stanford Cars, Flowers-102, or a custom dataset). Your project must include:

1. **Data Pipeline:** Custom Dataset class with train/val/test splits, data augmentation, and DataLoader with proper worker configuration.
2. **Model:** A fine-tuned pretrained model with a custom classification head.
3. **Training:**
   - Mixed precision training
   - Learning rate scheduling (cosine annealing or one-cycle)
   - Early stopping based on validation loss
   - Checkpoint saving (best model and latest model)
4. **Monitoring:** TensorBoard logging of losses, accuracy, learning rate, and sample predictions.
5. **Evaluation:**
   - Confusion matrix
   - Per-class precision, recall, and F1-score
   - Grad-CAM visualizations on 10 correctly and 5 incorrectly classified images
6. **Export:** Save the final model as TorchScript and ONNX. Write a standalone inference script that loads the exported model and classifies a single image from a file path.
7. **Report:** A README documenting your dataset, architecture choices, training decisions, final results, and lessons learned.

**Deliverables:** A GitHub-ready repository with organized code, a trained model checkpoint, export files, and a comprehensive README.

---

## Tips for Success

- **Don't skip assignments.** Each one builds on concepts from the previous ones.
- **Break when stuck, not when bored.** If something isn't working, add print statements, check tensor shapes, and read error messages carefully.
- **Read the PyTorch docs.** Every function mentioned above has excellent documentation at [pytorch.org/docs](https://pytorch.org/docs/stable/).
- **Use Google Colab** if you don't have a GPU — it's free and supports CUDA.
- **Version your code.** Use Git from the very first assignment. Your future self will thank you.
