import torch
from Tokenizers import TokenizersConfig, Tokenizers
from BEATs import BEATs, BEATsConfig


#################################  TOKENIZER  ####################################
# Takes the path to the checkpoint file as input.
# Loads the checkpoint, configures the tokenizer, and sets it to evaluation mode.
# Tokenization Method (tokenize):

# Accepts audio input and padding mask.
# Returns the tokenized label.
##################################################################################
class BEATsTokenizerWrapper:
    def __init__(self, checkpoint_path):
        """
        Initializes the BEATsTokenizerWrapper with a pre-trained tokenizer checkpoint.
        
        Parameters:
        - checkpoint_path (str): Path to the tokenizer checkpoint file.
        """
        # Load the pre-trained checkpoints
        checkpoint = torch.load(checkpoint_path)
        
        # Configure the tokenizer
        cfg = TokenizersConfig(checkpoint['cfg'])
        self.tokenizer = Tokenizers(cfg)
        self.tokenizer.load_state_dict(checkpoint['model'])
        self.tokenizer.eval()

    def tokenize(self, audio_input, padding_mask):
        """
        Tokenizes the audio input and generates the labels.
        
        Parameters:
        - audio_input (torch.Tensor): Tensor containing audio input.
        - padding_mask (torch.Tensor): Tensor containing padding mask.
        
        Returns:
        - labels (torch.Tensor): Tokenized labels.
        """
        # Ensure the model is in evaluation mode
        self.tokenizer.eval()
        
        # Generate labels
        labels = self.tokenizer.extract_labels(audio_input, padding_mask=padding_mask)
        return labels

# Example usage:
# tokenizer = BEATsTokenizerWrapper(r'F:\Github\NTU_BEATs\NTU_pretrain_BEATs\beats\Tokenizer_iter3_plus_AS2M.pt')
# audio_input_16khz = torch.randn(1, 10000)
# padding_mask = torch.zeros(1, 10000).bool()
# labels = tokenizer.tokenize(audio_input_16khz, padding_mask)


#################################  Loads Pre-trained BEATs  #########################
# Accepts the path to the checkpoint file.
# Loads the checkpoint, configures the BEATs model, and sets it to evaluation mode.
# Feature Extraction Method (extract_features):

# Takes audio input and padding mask as arguments.
# Returns the extracted audio representation.    
#####################################################################################    
class BEATsModelWrapper:
    def __init__(self, checkpoint_path):
        """
        Initializes the BEATsModelWrapper with a pre-trained BEATs model checkpoint.
        
        Parameters:
        - checkpoint_path (str): Path to the BEATs model checkpoint file.
        """
        # Load the pre-trained checkpoints
        checkpoint = torch.load(checkpoint_path)
        
        # Configure the model
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def extract_features(self, audio_input, padding_mask):
        """
        Extracts features from the audio input using the BEATs model.
        
        Parameters:
        - audio_input (torch.Tensor): Tensor containing audio input.
        - padding_mask (torch.Tensor): Tensor containing padding mask.
        
        Returns:
        - features (torch.Tensor): Extracted audio representation.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Extract features
        features = self.model.extract_features(audio_input, padding_mask=padding_mask)[0]
        return features

# Example usage:
# model_wrapper = BEATsModelWrapper(r'F:\Github\NTU_BEATs\NTU_pretrain_BEATs\beats\BEATs_iter3_plus_AS2M.pt')
# audio_input_16khz = torch.randn(1, 10000)
# padding_mask = torch.zeros(1, 10000).bool()
# features = model_wrapper.extract_features(audio_input_16khz, padding_mask)


##############################  Load Fine-tuned Models  #####################################
##Accepts the path to the checkpoint file.
# Loads the checkpoint, configures the BEATs model, and sets it to evaluation mode.
# Feature Extraction Method (extract_features):

# Takes audio input and padding mask as arguments.
# Returns the extracted audio representation.    
##############################################################################################
class BEATsClassifier:
    def __init__(self, checkpoint_path):
        """
        Initializes the BEATsClassifier with a fine-tuned BEATs model checkpoint.
        
        Parameters:
        - checkpoint_path (str): Path to the fine-tuned BEATs model checkpoint file.
        """
        # Load the fine-tuned checkpoints
        checkpoint = torch.load(checkpoint_path)
        
        # Configure the model
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # Load label dictionary
        self.label_dict = checkpoint['label_dict']

    def predict(self, audio_input, padding_mask):
        """
        Predicts the classification probability of each class for the given audio input.
        
        Parameters:
        - audio_input (torch.Tensor): Tensor containing audio input.
        - padding_mask (torch.Tensor): Tensor containing padding mask.
        
        Returns:
        - None: Prints the top 5 predicted labels with their probabilities.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Predict probabilities
        probs = self.model.extract_features(audio_input, padding_mask=padding_mask)[0]
        
        # Print top 5 predictions for each audio input
        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
            top5_label = [self.label_dict[label_idx.item()] for label_idx in top5_label_idx]
            print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')

# Example usage:
# classifier = BEATsClassifier('/path/to/model.pt')
# audio_input_16khz = torch.randn(3, 10000)
# padding_mask = torch.zeros(3, 10000).bool()
# classifier.predict(audio_input_16khz, padding_mask)