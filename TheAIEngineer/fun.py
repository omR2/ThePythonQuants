def relu(x):
  return x * (x > 0)

def relu_prime(x):
  return 1. * (x > 0)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
  return sigmoid(x) * (1 - sigmoid(x))

class TinyMLP(nn.Module):
  def __init__(self, d, h, Theta):
    super().__init__()
    # Initialize parameters from Theta
    self.W1 = nn.Parameter(torch.tensor(Theta['W1'], dtype=torch.float32))
    self.b1 = nn.Parameter(torch.tensor(Theta['b1'], dtype=torch.float32))
    self.W2 = nn.Parameter(torch.tensor(Theta['W2'], dtype=torch.float32))
    self.x = nn.Parameter(torch.tensor(Theta['x'], dtype=torch.float32)) # User's modification
    self.y = nn.Parameter(torch.tensor(Theta['y'], dtype=torch.float32)) # User's modification
    self.lr = nn.Parameter(torch.tensor(Theta['lr'], dtype=torch.float32)) # User's modification
    # Handle b2 as a scalar if it's a single-element list
    if isinstance(Theta['b2'], list) and len(Theta['b2']) == 1:
        self.b2 = nn.Parameter(torch.tensor(Theta['b2'][0], dtype=torch.float32))
    else:
        self.b2 = nn.Parameter(torch.tensor(Theta['b2'], dtype=torch.float32))

  def forward(self): # User's modification: no input x, uses self.x
    a1 = self.W1 @ self.x + self.b1
    h1 = torch.relu(a1)
    f = self.W2 @ h1 + self.b2
    # User's modification: returns a dictionary
    return {'f':f,  'bSigmoid': False, 'bTorch': True,
          'Theta': {'x': self.x,
          'W1': self.W1,
          'b1': self.b1,
          'W2': self.W2,
          'b2': self.b2},
          'a1':a1, 'h1':h1, 'y': self.y, 'lr': self.lr}           


def myMLPForward(Theta, bSigmoid = False, bTorch = 0):

  mySigmoid = sigmoid
  myRelu = relu
  x_in = Theta['x']
  W1_in = Theta['W1']
  b1_in = Theta['b1']
  W2_in = Theta['W2']
  b2_in = Theta['b2']
  y_val = Theta['y']
  lr_val = Theta['lr']

  if bTorch == 0:
    x = np.array(x_in).T
    W1 = np.array(W1_in)
    b1 = np.array(b1_in).T
    W2 = np.array(W2_in)
    b2 = np.array(b2_in)
  else: # bTorch is 1 or -1 (PyTorch mode)
    # Clone tensors to ensure independent graphs for each forward pass
    x = torch.tensor(x_in, dtype=torch.float32).clone().T
    W1 = torch.tensor(W1_in, dtype=torch.float32).clone().requires_grad_(bTorch==1)
    b1 = torch.tensor(b1_in, dtype=torch.float32).clone().requires_grad_(bTorch==1)
    W2 = torch.tensor(W2_in, dtype=torch.float32).clone().requires_grad_(bTorch==1)
    b2 = torch.tensor(b2_in, dtype=torch.float32).clone().requires_grad_(bTorch==1)
    mySigmoid = torch.sigmoid
    myRelu = torch.relu

  a1 = W1 @ x + b1
  if bSigmoid == True:
    h1 = mySigmoid(a1)
  else:
    h1 = myRelu(a1)
  f = W2 @ h1 + b2
  # Return the processed tensors/arrays along with other info, including y and lr
  return {'f':f,  'bSigmoid': bSigmoid, 'bTorch': bTorch,
          'Theta': {'x': x,
          'W1': W1,
          'b1': b1,
          'W2': W2,
          'b2': b2},
          'a1':a1, 'h1':h1, 'y': y_val, 'lr': lr_val}
          
def myBuiltIn(d,h,Theta):
  model = nn.Sequential(
  nn.Linear(2, 2),
  nn.ReLU(),
  nn.Linear(2, 1)
  )

  # Set initial parameters from myPars
  # For the first linear layer (index 0)
  model[0].weight = nn.Parameter(torch.tensor(Theta['W1'], dtype=torch.float32))
  model[0].bias = nn.Parameter(torch.tensor(Theta['b1'], dtype=torch.float32))

  # For the second linear layer (index 2)
  model[2].weight = nn.Parameter(torch.tensor(Theta['W2'], dtype=torch.float32))
  # Handle b2 as a scalar if it's a single-element list in myPars
  if isinstance(Theta['b2'], list) and len(Theta['b2']) == 1:
      model[2].bias = nn.Parameter(torch.tensor(Theta['b2'][0], dtype=torch.float32))
  else:
      model[2].bias = nn.Parameter(torch.tensor(Theta['b2'], dtype=torch.float32))

  return model    
def myMLPBackward(theta):

  f = theta['f']
  a1 = theta['a1']
  h1 = theta['h1']
  bSigmoid = theta['bSigmoid']
  bTorch = theta['bTorch']

  # Retrieve processed x, W1, b1, W2, b2, y, lr from the dictionary
  x = theta['Theta']['x']
  W1 = theta['Theta']['W1']
  b1 = theta['Theta']['b1']
  W2 = theta['Theta']['W2']
  b2 = theta['Theta']['b2']
  y = theta['y']
  lr = theta['lr']

  L = 0.5 * (f - y)**2

  grad_norms = {}

  if bTorch == 1:
    # Clear previous gradients
    if W1.grad is not None: W1.grad.zero_()
    if b1.grad is not None: b1.grad.zero_()
    if W2.grad is not None: W2.grad.zero_()
    if b2.grad is not None: b2.grad.zero_()

    # Only call backward if requires_grad is True for any parameter
    # and the tensor `f` is part of a graph that can be differentiated.
    if f.requires_grad:
        L.backward() # This will compute gradients

    # Collect gradient norms AFTER backward pass
    grad_norms['W1'] = W1.grad.norm().item() if W1.grad is not None else 0.0
    grad_norms['b1'] = b1.grad.norm().item() if b1.grad is not None else 0.0
    grad_norms['W2'] = W2.grad.norm().item() if W2.grad is not None else 0.0
    grad_norms['b2'] = b2.grad.norm().item() if b2.grad is not None else 0.0

    # Use an optimizer to update parameters
    optimizable_params = []
    if W1.requires_grad: optimizable_params.append(W1)
    if b1.requires_grad: optimizable_params.append(b1)
    if W2.requires_grad: optimizable_params.append(W2)
    if b2.requires_grad: optimizable_params.append(b2)

    optimizer = torch.optim.SGD(optimizable_params, lr=lr)
    optimizer.step() # Apply the updates

  else:
    # backward (manual calculation for NumPy or PyTorch without grad)
    df = (f - y)
    dW2 = df * h1[None, :] # (1,h)
    db2 = df
    dh1 = W2.T * df # (h,)
    # h2 must be defined before da1 calculation
    if bSigmoid == True:
      h2 = sigmoid_prime(a1)
    else:
      h2 = relu_prime(a1)
    da1 = dh1.squeeze() * h2
    dW1 = da1[:,None] @ x[None,:]
    db1 = da1

    # Collect gradient norms BEFORE SGD update for NumPy
    grad_norms['W1'] = np.linalg.norm(dW1) if isinstance(dW1, np.ndarray) else dW1.norm().item()
    grad_norms['b1'] = np.linalg.norm(db1) if isinstance(db1, np.ndarray) else db1.norm().item()
    grad_norms['W2'] = np.linalg.norm(dW2) if isinstance(dW2, np.ndarray) else dW2.norm().item()
    grad_norms['b2'] = np.linalg.norm(db2) if isinstance(db2, np.ndarray) else db2.norm().item()

    # SGD update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1 # Corrected from dW1 to db1

  # Return the updated Theta dictionary AND grad_norms
  return {'f':f, 'bSigmoid': bSigmoid, 'bTorch': bTorch,
          'Theta': {'x': x,
          'W1': W1,
          'b1': b1,
          'W2': W2,
          'b2': b2},
          'a1':a1, 'h1':h1, 'y': y, 'lr': lr, 'grad_norms': grad_norms}


def run_training_simulation(num_iterations, d, h, initial_pars, bSigmoid, bTorch_mode):
  losses = []
  grad_norms = {'W1': [], 'b1': [], 'W2': [], 'b2': []}
  h1_sums = []

  # Initialize parameters for this run (deep copy to ensure independence)
  current_pars = {key: (value.copy() if isinstance(value, list) or isinstance(value, np.ndarray) else value) for key, value in initial_pars.items()}

  # Special handling for TinyMLP which needs a model instance
  if bTorch_mode == 2: # Representing TinyMLP
    model = TinyMLP(d, h, Theta=current_pars)
    optimizer = torch.optim.SGD(model.parameters(), lr=current_pars['lr'])

  for i in range(num_iterations):
    if bTorch_mode == 2:
      # TinyMLP forward pass
      output_dict = model.forward()
      f_val = output_dict['f']
      y_val = output_dict['y']
      h1_val = output_dict['h1']

      L = 0.5 * (f_val - y_val)**2
      losses.append(L.item())

      optimizer.zero_grad()
      L.backward()

      # Collect grad norms for TinyMLP
      grad_norms['W1'].append(model.W1.grad.norm().item() if model.W1.grad is not None else 0.0)
      grad_norms['b1'].append(model.b1.grad.norm().item() if model.b1.grad is not None else 0.0)
      grad_norms['W2'].append(model.W2.grad.norm().item() if model.W2.grad is not None else 0.0)
      grad_norms['b2'].append(model.b2.grad.norm().item() if model.b2.grad is not None else 0.0)

      optimizer.step()
      h1_sums.append(h1_val.sum().item())

    else:
      # myMLPForward pass
      output_fwd = myMLPForward(bSigmoid=bSigmoid, bTorch=bTorch_mode, Theta=current_pars)
      f_val = output_fwd['f']
      h1_val = output_fwd['h1']
      y_val = output_fwd['y']
      lr_val = output_fwd['lr']

      # Create a dictionary compatible with myMLPBackward's input structure
      backward_input = {
          'f': f_val,
          'a1': output_fwd['a1'],
          'h1': h1_val,
          'bSigmoid': bSigmoid,
          'bTorch': bTorch_mode,
          'Theta': output_fwd['Theta'],
          'y': y_val,
          'lr': lr_val
      }

      # myMLPBackward pass
      result_bwd = myMLPBackward(backward_input)
      L = 0.5 * (result_bwd['f'] - result_bwd['y'])**2
      losses.append(L.item() if isinstance(L, torch.Tensor) else L.item())

      # Update current_pars with the *updated* parameters from result_bwd
      for param_key in ['x', 'W1', 'b1', 'W2', 'b2']:
        current_pars[param_key] = result_bwd['Theta'][param_key]

      # Collect gradient norms
      for gk in grad_norms.keys():
        grad_norms[gk].append(result_bwd['grad_norms'][gk])

      # Collect h1 sum
      h1_sums.append(h1_val.sum().item() if isinstance(h1_val, torch.Tensor) else h1_val.sum())

  return {'losses': losses, 'grad_norms': grad_norms, 'h1_sums': h1_sums}

def run_sequential_training_simulation(num_iterations, d, h, initial_pars, model):
  losses = []
  grad_norms = {'W1': [], 'b1': [], 'W2': [], 'b2': []}
  h1_sums = []

  # The input 'x' and target 'y' are not model parameters; they are inputs to the forward pass
  # We need to make sure they are tensors for the model
  x_tensor = torch.tensor(initial_pars['x'], dtype=torch.float32)
  y_tensor = torch.tensor(initial_pars['y'], dtype=torch.float32)
  lr = initial_pars['lr']

  optimizer = torch.optim.SGD(model.parameters(), lr=lr)

  for i in range(num_iterations):
    # Forward pass
    f_val = model(x_tensor)
    
    # Calculate loss
    L = 0.5 * (f_val - y_tensor)**2
    losses.append(L.item()) # Store loss value

    # Backward pass and optimization
    optimizer.zero_grad()
    L.backward()

    # Collect grad norms
    grad_norms['W1'].append(model[0].weight.grad.norm().item() if model[0].weight.grad is not None else 0.0)
    grad_norms['b1'].append(model[0].bias.grad.norm().item() if model[0].bias.grad is not None else 0.0)
    grad_norms['W2'].append(model[2].weight.grad.norm().item() if model[2].weight.grad is not None else 0.0)
    grad_norms['b2'].append(model[2].bias.grad.norm().item() if model[2].bias.grad is not None else 0.0)

    optimizer.step()

    # Get hidden layer activity (h1) AFTER update
    with torch.no_grad(): # Don't track gradients for this intermediate value
        a1_val = model[0](x_tensor) # First linear layer output (a1)
        h1_val = model[1](a1_val)    # ReLU activation (h1)
        h1_sums.append(h1_val.sum().item())

  return {'losses': losses, 'grad_norms': grad_norms, 'h1_sums': h1_sums}
  

def plot_simulation_results(results, title_suffix=""):
  fig, axs = plt.subplots(3, 1, figsize=(10, 15))

  # Plot Loss
  axs[0].plot(results['losses'], label=f'Loss {title_suffix}')
  axs[0].set_title(f'Loss over Iterations {title_suffix}')
  axs[0].set_xlabel('Iteration')
  axs[0].set_ylabel('Loss')
  axs[0].grid(True)
  axs[0].legend()

  # Plot Gradient Norms
  for param_name, norms in results['grad_norms'].items():
    axs[1].plot(norms, label=f'{param_name} Grad Norm {title_suffix}')
  axs[1].set_title(f'Gradient Norms over Iterations {title_suffix}')
  axs[1].set_xlabel('Iteration')
  axs[1].set_ylabel('L2 Norm of Gradient')
  axs[1].grid(True)
  axs[1].legend()

  # Plot Hidden Layer Activity (Sum of h1)
  axs[2].plot(results['h1_sums'], label=f'Sum of h1 {title_suffix}')
  axs[2].set_title(f'Sum of Hidden Layer (h1) Activity over Iterations {title_suffix}')
  axs[2].set_xlabel('Iteration')
  axs[2].set_ylabel('Sum of h1')
  axs[2].grid(True)
  axs[2].legend()

  plt.tight_layout()
  plt.show()
          
def plot_comparison_metrics(simulation_results_list, figure_title):
  fig, axs = plt.subplots(6, 1, figsize=(12, 30))
  fig.suptitle(figure_title, fontsize=16)

  # Plot 1: Losses
  for res_entry in simulation_results_list:
    axs[0].plot(res_entry['results']['losses'], label=res_entry['label'])
  axs[0].set_title('Loss over Iterations')
  axs[0].set_xlabel('Iteration')
  axs[0].set_ylabel('Loss')
  axs[0].grid(True)
  axs[0].legend()

  # Plot 2-5: Gradient Norms (W1, b1, W2, b2)
  grad_param_names = ['W1', 'b1', 'W2', 'b2']
  for i, param_name in enumerate(grad_param_names):
    for res_entry in simulation_results_list:
      axs[i + 1].plot(res_entry['results']['grad_norms'][param_name], label=res_entry['label'])
    axs[i + 1].set_title(f'{param_name} Gradient Norm over Iterations')
    axs[i + 1].set_xlabel('Iteration')
    axs[i + 1].set_ylabel('L2 Norm of Gradient')
    axs[i + 1].grid(True)
    axs[i + 1].legend()

  # Plot 6: Hidden Layer Activity (Sum of h1)
  for res_entry in simulation_results_list:
    axs[5].plot(res_entry['results']['h1_sums'], label=res_entry['label'])
  axs[5].set_title('Sum of Hidden Layer (h1) Activity over Iterations')
  axs[5].set_xlabel('Iteration')
  axs[5].set_ylabel('Sum of h1')
  axs[5].grid(True)
  axs[5].legend()

  plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap
  plt.show()          
          