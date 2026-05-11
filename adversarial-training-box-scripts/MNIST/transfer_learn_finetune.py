from random import shuffle
from adversarial_training_box.models.MNIST.cnn_yang_big import CNN_YANG_BIG
import copy
import torch
import torch.optim as optim
import torchvision
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

def reset_last_k_layers(model, k):
    """Reset the parameters of the last k layers of a model"""
    layers = list(model.children())

    if k > len(layers):
        raise ValueError(f"k ({k}) cannot be larger than the number of layers ({len(layers)})")
    
    if k <= 0:
        return  # Nothing to reset
    
    # Reset the last k layers
    for layer in layers[-k:]:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    # TODO: investigate for other cases!

def scale_except_last_k_layers(model, k, scale_factor = 0.001):
    """Freeze all parameters except the last k layers"""
    model.requires_grad_(True)

    layers = list(model.children())
    
    if k > len(layers):
        raise ValueError(f"k ({k}) cannot be larger than the number of layers ({len(layers)})")

    # Scale all layers except the last k
    for layer in layers[:-k]:
        for param in layer.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: grad * scale_factor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning script with configurable network and experiment name')
    parser.add_argument('--source_model_path', type=str, default='"generated/BachelorThesisRuns/cnn_yang_big-pgd-training_21-10-2025+12_40/cnn_yang_big.pth"',
                       help='Source model path')
    parser.add_argument('--experiment_name', type=str, default="DefaultExperiment",
                       help='Custom experiment name (default: {source_model_path}-transfer-learning)')
    parser.add_argument('--retraining_layers', type=int, default=1,
                       help='Indicate number of layers to retrain')
    args = parser.parse_args() 

    training_parameters = AttributeDict(
        learning_rate = 0.001,
        weight_decay = 1e-4,
        scheduler_step_size=10,
        scheduler_gamma=0.98,
        attack_epsilon=0.3, 
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
    def convert_last_layer(network, num_classes=47):
        layers = list(network.named_modules())
        last_layer_name = layers[-1][0]
        last_layer_module = layers[-1][1]

        in_features = last_layer_module.in_features
        has_bias = last_layer_module.bias is not None
        new_layer = torch.nn.Linear(in_features, num_classes, bias=has_bias)

        # Replace the last layer in the network
        setattr(network, last_layer_name, new_layer)
        
        return network

    converted_model = convert_last_layer(source_model_copy)
    
    # reset_last_k_layers(converted_model, retraining_layers)

    scale_except_last_k_layers(converted_model, retraining_layers)

    # Training configuration
    optimizer = getattr(optim, 'Adam')(converted_model.parameters(), lr=training_parameters.learning_rate, weight_decay=training_parameters.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_parameters.scheduler_step_size, gamma=training_parameters.scheduler_gamma)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=training_parameters.patience_epochs, delta=training_parameters.overhead_delta)

    # Train, validation and test dataset
    dataset = torchvision.datasets.EMNIST('../data', split="balanced",train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset,validation_dataset, = torch.utils.data.random_split(dataset, (0.8, 0.2))
    test_dataset = torchvision.datasets.EMNIST('../data', split="balanced",train=False, download=True, transform=torchvision.transforms.ToTensor())

   # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # Validation module
    validation_module = StandardTestModule(criterion=criterion)

    # Training modules stack
    training_stack = []
    training_stack.append((50, StandardTrainingModule(criterion=criterion)))

    # Testing modules stack
    testing_stack = [
        StandardTestModule(),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.1),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.2),
        StandardTestModule(attack=FGSMAttack(), epsilon=0.3),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.1),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.2),
        StandardTestModule(attack=PGDAttack(epsilon_step_size=0.01, number_iterations=40, random_init=True), epsilon=0.3),
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
    experiment_tracker.initialize_new_experiment(f"TL{retraining_layers}_finetune", training_parameters=training_parameters | training_objects)
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
