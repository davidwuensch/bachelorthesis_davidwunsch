from random import shuffle
import copy
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.nn as nn
from pathlib import Path
import optuna
import argparse
from optuna.trial import TrialState
from adversarial_training_box.pipeline.early_stopper import EarlyStopper

from adversarial_training_box.adversarial_attack.pgd_attack import PGDAttack
from adversarial_training_box.adversarial_attack.fgsm_attack import FGSMAttack
from adversarial_training_box.database.experiment_tracker import ExperimentTracker
from adversarial_training_box.database.attribute_dict import AttributeDict
from adversarial_training_box.pipeline.pipeline import Pipeline
from adversarial_training_box.models.emnist_net_256x2 import EMNIST_NET_256x2
from adversarial_training_box.pipeline.standard_training_module import StandardTrainingModule
from adversarial_training_box.pipeline.standard_test_module import StandardTestModule
from adversarial_training_box.adversarial_attack.auto_attack_module import AutoAttackModule
from torchvision.models.resnet import BasicBlock, Bottleneck

def get_resnet_blocks(model):
    """Extract all ResNet blocks"""

    blocks = []
    # For the ResNet architecture, that we use
    if hasattr(model, 'conv2_x'):
        blocks.extend([
            ('conv1', model.conv1),
            ('conv2_x', model.conv2_x),
            ('conv3_x', model.conv3_x),
            ('conv4_x', model.conv4_x),
            ('conv5_x', model.conv5_x),
            ('avg_pool', model.avg_pool),
            ('fc', model.fc)
        ])
    else:
        # Fallback for other ResNet structures
        for name, module in model.named_children():
            blocks.append((name, module))
    return blocks

def reset_last_k_layers(model, k):
    """Reset the parameters of the last k layers of a model"""
    blocks = get_resnet_blocks(model)

    if k > len(blocks):
        raise ValueError(f"k ({k}) cannot be larger than the number of layers ({len(blocks)})")
    
    if k <= 0:
        raise ValueError(f"k must be positive (got {k}). For transfer learning, you must retrain at least 1 block.")
    
    # Reset the last k blocks
    for name, block in blocks[-k:]:
        print(f"  Resetting: {name}")
        for module in block.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

def freeze_except_last_k_layers(model, k):
    """Freeze all parameters except the last k layers"""
    blocks = get_resnet_blocks(model)

    if k > len(blocks):
        raise ValueError(f"k ({k}) cannot be larger than the number of layers ({len(blocks)})")
    
    if k <= 0:
        raise ValueError(f"k must be positive (got {k}). For transfer learning, you must retrain at least 1 block.")

    for param in model.parameters():
        param.requires_grad= False

    # Unfreeze only the last k blocks
    for name, block in blocks[-k:]:
        print(f"  Unfreezing: {name}")
        for param in block.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Transfer Learning script with configurable network and experiment name')
    parser.add_argument('--source_model_path', type=str, default='"generated/BachelorThesisRuns/cnn_yang_big-pgd-training_21-10-2025+12_40/cnn_yang_big.pth"',
                       help='Source model path')
    parser.add_argument('--experiment_name', type=str, default="DefaultExperiment",
                       help='Custom experiment name (default: {source_model_path}-transfer-learning)')
    parser.add_argument('--retraining_layers', type=int, default=1,
                       help='Indicate number of layers to retrain')
    args = parser.parse_args()

    training_parameters = AttributeDict(
        learning_rate = 0.1,
        weight_decay = 5e-4,
        momentum = 0.9,
        scheduler_milestones=[60, 120, 160],
        scheduler_gamma=0.2,
        patience_epochs=6,
        overhead_delta=0.0,
        batch_size=256)
    
    source_model_path = Path(args.source_model_path)
    experiment_name = args.experiment_name
    retraining_layers = args.retraining_layers

    # Source model
    source_model = torch.load(source_model_path, map_location='cpu')
    source_model_copy = copy.deepcopy(source_model)

    # Network converter to adapt to target domain
    def convert_last_layer(network, num_classes=100, hidden_size = 512):
        layers = list(network.named_modules())
        last_layer_name = layers[-1][0]
        last_layer_module = layers[-1][1]

        in_features = last_layer_module.in_features
        has_bias = last_layer_module.bias is not None

        # Create a small MLP with two hidden layers
        new_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size,bias=True), 
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),                         
            nn.Dropout(0.25),       
            
            nn.Linear(hidden_size, hidden_size, bias=True), # Input and output size are `hidden_size`
            nn.BatchNorm1d(hidden_size), 
            nn.ReLU(),                         
            nn.Dropout(0.25),

            nn.Linear(hidden_size, hidden_size, bias=True), # Input and output size are `hidden_size`
            nn.BatchNorm1d(hidden_size), 
            nn.ReLU(),                         
            nn.Dropout(0.25),

            nn.Linear(hidden_size, num_classes,bias=True) 
        )

        # Replace the last layer in the network
        setattr(network, last_layer_name, new_mlp)
        
        return network

    converted_model = convert_last_layer(source_model_copy)
    
    reset_last_k_layers(converted_model, retraining_layers)

    freeze_except_last_k_layers(converted_model, retraining_layers)

    cifar_mean = [0.5071, 0.4865, 0.4409]
    cifar_std = [0.2673, 0.2564, 0.2762]
    
    normalize = transforms.Normalize(mean=cifar_mean,std=cifar_std)

    mean_std = sum(cifar_std) / len(cifar_std)

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    num_classes = 100
    full_train_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)
    full_validation_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=test_transform)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(43)

    # Split with same indices for both
    train_dataset, _ = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, validation_dataset = torch.utils.data.random_split(full_validation_dataset, [train_size, val_size], generator=generator)

    test_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=test_transform)
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)    

    # Training configuration
    optimizer = getattr(optim, 'SGD')(converted_model.parameters(), lr=training_parameters.learning_rate, weight_decay=training_parameters.weight_decay, momentum=training_parameters.momentum)
    scheduler = MultiStepLR(optimizer, milestones=training_parameters.scheduler_milestones, gamma=training_parameters.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=training_parameters.patience_epochs, delta=training_parameters.overhead_delta)

    # Validation module
    validation_module = StandardTestModule(criterion=criterion)

    # Training modules stack
    training_stack = []
    training_stack.append((200, StandardTrainingModule(criterion=criterion)))

    # Testing modules stack
    testing_stack = [StandardTestModule(),
        StandardTestModule(attack=FGSMAttack(), epsilon=2/255/mean_std),
        StandardTestModule(attack=FGSMAttack(), epsilon=4/255/mean_std),
        StandardTestModule(attack=FGSMAttack(), epsilon=8/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=2/255/mean_std/4, number_iterations=20, random_init=True), epsilon=2/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=4/255/mean_std/4, number_iterations=20, random_init=True), epsilon=4/255/mean_std),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=8/255/mean_std/4, number_iterations=20, random_init=True), epsilon=8/255/mean_std),
    ]
    
    # Convert complex objects to JSON-serializable format
    def serialize_training_stack(stack):
        return [{"epochs": epochs, "module_type": type(module).__name__, 
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for epochs, module in stack]
    
    def serialize_testing_stack(stack):
        return [{"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)} for module in stack]
    
    def serialize_validation_module(module):
        return {"module_type": type(module).__name__,
                "attack": type(getattr(module, 'attack', None)).__name__ if hasattr(module, 'attack') and module.attack else "None",
                "epsilon": getattr(module, 'epsilon', None)}

    training_objects = AttributeDict(criterion=str(criterion), 
                                     optimizer=str(optimizer), 
                                     network=str(converted_model), 
                                     scheduler=str(scheduler), 
                                     training_stack=serialize_training_stack(training_stack),
                                     testing_stack=serialize_testing_stack(testing_stack),
                                     validation_module=serialize_validation_module(validation_module))

    # Setup experiment
    experiment_tracker = ExperimentTracker(experiment_name, Path("./generated"), login=True)
    experiment_tracker.initialize_new_experiment(f"TL{retraining_layers}_512MLP3BatchDropout", training_parameters=training_parameters | training_objects)
    pipeline = Pipeline(experiment_tracker, training_parameters, criterion, optimizer, scheduler)
    
    # Train
    pipeline.train(train_loader, converted_model, training_stack, early_stopper=early_stopper, 
                   validation_loader=validation_loader,
                   validation_module=validation_module, retraining_layers=retraining_layers
                   )
    
    # Test
    network = experiment_tracker.load_trained_model(converted_model)
    pipeline.test(network, test_loader, testing_stack=testing_stack)
    experiment_tracker.export_to_onnx(network, test_loader)

    # Finish logging
    experiment_tracker.finish()