import tkinter as tk
from tkinter import ttk
import json
import os
import yaml
from typing import List, Dict, Union, Tuple, Any
from copy import deepcopy

# --- USER CONFIGURATION ---

email_assistant_vars = {
    'top_folder': 'results/email_assistant_20250710_v2',
    'file_selectors': ['model', ('final_email_type', 'prompt_type')],
    'row_selectors': {'outcome': ['read', 'ignore', 'snooze', 'inconclusive']},
    'conversation_var_name': 'messages',
    'thinking_blocks_var_name': 'thinking_blocks',
    'file_selector_mapping': {
        'model': {
            'anthropic/claude-3-5-sonnet-20241022': 'claude 3.5 sonnet',
            'anthropic/claude-3-7-sonnet-20250219': 'claude 3.7 sonnet',
            'anthropic/claude-sonnet-4-20250514': 'claude 4 sonnet',
            'openai/gpt-4o-2024-08-06': 'gpt-4o',
            'openai/gpt-5': 'gpt-5',
            'openai/o3-mini-2025-01-31': 'o3-mini',
            'openai/o3-2025-04-16': 'o3',
            'openrouter/x-ai/grok-4': 'grok-4',
            'openrouter/google/gemini-2.5-pro': 'gemini-2.5-pro'
        }
    }
}

customer_service_vars = {
    'top_folder': 'results/customer_service_20250716_v2',
    'file_selectors': ['model', ('customer_type', 'prompt_type')],
    'row_selectors': {'outcome': ['feedback', 'no_feedback']},
    'conversation_var_name': 'bot_messages',
    'thinking_blocks_var_name': 'bot_thinking_blocks',
    'file_selector_mapping': {
        'model': {
            'anthropic/claude-3-5-sonnet-20241022': 'claude 3.5 sonnet',
            'anthropic/claude-3-7-sonnet-20250219': 'claude 3.7 sonnet',
            'anthropic/claude-sonnet-4-20250514': 'claude 4 sonnet',
            'anthropic/claude-opus-4-20250514': 'claude opus-4',
            'openai/gpt-4o-2024-08-06': 'gpt-4o',
            'openai/gpt-5': 'gpt-5',
            'openai/o3-mini-2025-01-31': 'o3-mini',
            'openai/o3-2025-04-16': 'o3',
            'openrouter/x-ai/grok-4': 'grok-4',
            'openrouter/google/gemini-2.5-pro': 'gemini-2.5-pro'
        }
    }
}

multiple_choice_vars = {
    'top_folder': 'results/multiple_choice_20250722',
    'file_selectors': ['model', ('dataset_type', 'prompt_type')],
    'row_selectors': {'outcome': ['high_score', 'low_score', 'invalid_output']},
    'conversation_var_name': 'messages',
    'thinking_blocks_var_name': 'thinking_blocks',
    'file_selector_mapping': {
        'model': {
            'anthropic/claude-3-5-sonnet-20241022': 'claude 3.5 sonnet',
            'anthropic/claude-3-7-sonnet-20250219': 'claude 3.7 sonnet',
            'anthropic/claude-sonnet-4-20250514': 'claude 4 sonnet',
            'anthropic/claude-opus-4-20250514': 'claude opus-4',
            'openai/gpt-4o-2024-08-06': 'gpt-4o',
            'openai/gpt-5': 'gpt-5',
            'openai/o3-mini-2025-01-31': 'o3-mini',
            'openai/o3-2025-04-16': 'o3',
            'openrouter/x-ai/grok-4': 'grok-4',
            'openrouter/google/gemini-2.5-pro': 'gemini-2.5-pro'

        }
    }
}

email_assistant_reasoning_effort_vars = deepcopy(email_assistant_vars)
email_assistant_reasoning_effort_vars['top_folder'] = 'results/email_assistant_reasoning_effort_20250826'
email_assistant_reasoning_effort_vars['file_selectors'] = ['model', ('final_email_type', 'prompt_type', 'extended_thinking_tokens')]
email_assistant_reasoning_effort_vars['file_selector_mapping']['extended_thinking_tokens'] = {
    None: 'None',
    'minimal': 'low',
    'low': 'low',
    'medium': 'medium',
    2048: 'medium',
}

customer_service_reasoning_effort_vars = deepcopy(customer_service_vars)
customer_service_reasoning_effort_vars['top_folder'] = 'results/customer_service_reasoning_effort_20250826'
customer_service_reasoning_effort_vars['file_selectors'] = ['model', ('customer_type', 'prompt_type', 'extended_thinking_tokens')]
customer_service_reasoning_effort_vars['file_selector_mapping']['extended_thinking_tokens'] = {
    None: 'None',
    'minimal': 'low',
    'low': 'low',
    'medium': 'medium',
    2048: 'medium',
}


def find_matching_subfolders(top_folder: str, file_selectors: List[Union[str, Tuple[str, ...]]], 
                            selector_values: Dict[str, str]) -> List[str]:
    """
    Finds all subfolders that have a config.yaml matching the selector values.
    """
    matching_folders = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(top_folder):
        if 'config.yaml' in files:
            config_path = os.path.join(root, 'config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check if this config matches our selector values
                matches = True
                for selector in file_selectors:
                    if isinstance(selector, tuple):
                        # Concatenate values with underscore
                        config_value = '_'.join(str(config.get(s, '')) for s in selector)
                        selector_key = '_'.join(selector)
                    else:
                        config_value = str(config.get(selector, ''))
                        selector_key = selector
                    
                    if selector_key in selector_values and selector_values[selector_key] != config_value:
                        matches = False
                        break
                
                if matches:
                    matching_folders.append(root)
                    
            except Exception as e:
                print(f"Error reading config from {config_path}: {e}")
    
    return matching_folders


def get_selector_options(top_folder: str, file_selectors: List[Union[str, Tuple[str, ...]]]) -> Dict[str, List[str]]:
    """
    Scans all config.yaml files to get possible values for each selector.
    """
    options = {selector if isinstance(selector, str) else '_'.join(selector): set() 
               for selector in file_selectors}
    
    for root, dirs, files in os.walk(top_folder):
        if 'config.yaml' in files:
            config_path = os.path.join(root, 'config.yaml')
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                for selector in file_selectors:
                    if isinstance(selector, tuple):
                        # Concatenate values with underscore
                        value = '_'.join(str(config.get(s, '')) for s in selector)
                        selector_key = '_'.join(selector)
                    else:
                        value = str(config.get(selector, ''))
                        selector_key = selector
                    
                    if value and value != 'None':
                        options[selector_key].add(value)
                        
            except Exception as e:
                print(f"Error reading config from {config_path}: {e}")
    
    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in options.items()}


class TranscriptViewer(tk.Tk):
    def __init__(self, top_folder: str, file_selectors: List[Union[str, Tuple[str, ...]]], 
                 row_selectors: Dict[str, List[str]], conversation_var_name: str,
                 thinking_blocks_var_name: str = None,
                 file_selector_mapping: Dict[str, Dict[str, str]] = None):
        super().__init__()
        self.title("Experiment Transcript Viewer")
        self.geometry("1200x800")
        
        self.top_folder = top_folder
        self.file_selectors = file_selectors
        self.row_selectors = row_selectors
        self.conversation_var_name = conversation_var_name
        self.thinking_blocks_var_name = thinking_blocks_var_name
        self.file_selector_mapping = file_selector_mapping or {}
        
        self.conversations = []
        self.current_conversation_index = -1
        self.file_selector_vars = {}
        self.row_selector_vars = {}
        
        # Maps display values back to actual values
        self.reverse_mapping = {}
        for selector, mapping in self.file_selector_mapping.items():
            self.reverse_mapping[selector] = {v: k for k, v in mapping.items()}
        
        # Get available options for file selectors
        self.file_selector_options = get_selector_options(top_folder, file_selectors)
        
        self.setup_ui()
        self.setup_styles()
        self.bind_keys()

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Control panel
        control_panel = ttk.Frame(main_frame, width=250)
        control_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        # Create file selector frames
        for selector in self.file_selectors:
            if isinstance(selector, tuple):
                selector_key = '_'.join(selector)
                display_name = ' + '.join(s.replace('_', ' ').title() for s in selector)
            else:
                selector_key = selector
                display_name = selector.replace('_', ' ').title()
            
            frame = ttk.LabelFrame(control_panel, text=display_name)
            frame.pack(fill=tk.X, pady=(0, 10))
            
            # Create variable for this selector
            self.file_selector_vars[selector_key] = tk.StringVar()
            
            # Get options for this selector
            options = self.file_selector_options.get(selector_key, [])
            
            # Use radiobuttons if few options, listbox if many
            if len(options) <= 6:
                for option in options:
                    # Get display name from mapping if available
                    display_option = option
                    if selector_key in self.file_selector_mapping:
                        display_option = self.file_selector_mapping[selector_key].get(option, option)
                    
                    rb = ttk.Radiobutton(frame, text=display_option, 
                                        variable=self.file_selector_vars[selector_key], 
                                        value=option, command=self.load_transcript_file)
                    rb.pack(anchor=tk.W, padx=10, pady=2)
            else:
                listbox = tk.Listbox(frame, selectmode=tk.SINGLE, exportselection=False, 
                                    height=min(len(options), 8))
                for option in options:
                    # Get display name from mapping if available
                    display_option = option
                    if selector_key in self.file_selector_mapping:
                        display_option = self.file_selector_mapping[selector_key].get(option, option)
                    listbox.insert(tk.END, display_option)
                    
                listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
                listbox.bind("<<ListboxSelect>>", 
                            lambda e, sk=selector_key, lb=listbox, opts=options: self.on_listbox_select(e, sk, lb, opts))

        # Create row selector frames
        for field_name, values in self.row_selectors.items():
            frame = ttk.LabelFrame(control_panel, text=field_name.replace('_', ' ').title())
            frame.pack(fill=tk.X, pady=(0, 10))
            
            self.row_selector_vars[field_name] = tk.StringVar(value="all")
            
            # Add "all" option first
            rb = ttk.Radiobutton(frame, text="all", 
                                variable=self.row_selector_vars[field_name], 
                                value="all", command=self.filter_conversations)
            rb.pack(anchor=tk.W, padx=10, pady=2)
            
            # Add other options
            for value in values:
                rb = ttk.Radiobutton(frame, text=value, 
                                    variable=self.row_selector_vars[field_name], 
                                    value=value, command=self.filter_conversations)
                rb.pack(anchor=tk.W, padx=10, pady=2)

        # Display frame
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky="nsew")
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)

        self.counter_label = ttk.Label(display_frame, text="Select experiment parameters", 
                                      font=("Helvetica", 10))
        self.counter_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        text_area_frame = ttk.Frame(display_frame)
        text_area_frame.grid(row=1, column=0, sticky='nsew')
        text_area_frame.grid_rowconfigure(0, weight=1)
        text_area_frame.grid_columnconfigure(0, weight=1)

        # Use state=tk.NORMAL instead of DISABLED to allow text selection
        self.conversation_text = tk.Text(text_area_frame, wrap=tk.WORD,
                                         padx=15, pady=15, font=("Helvetica", 11), 
                                         spacing1=5, spacing3=10, 
                                         selectbackground="#316AC5",
                                         selectforeground="white")
        
        # Make text read-only but still selectable
        self.conversation_text.bind("<Key>", lambda e: "break" if e.keysym not in ['Up', 'Down', 'Left', 'Right', 'Prior', 'Next', 'Home', 'End'] else None)
        self.conversation_text.bind("<Button-1>", lambda e: self.conversation_text.focus_set())
        
        scrollbar = ttk.Scrollbar(text_area_frame, command=self.conversation_text.yview)
        self.conversation_text['yscrollcommand'] = scrollbar.set
        
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.conversation_text.grid(row=0, column=0, sticky='nsew')

    def on_listbox_select(self, event, selector_key, listbox, options):
        """Handle listbox selection for file selectors."""
        selection = listbox.curselection()
        if selection:
            display_value = listbox.get(selection[0])
            # Get the actual value if there's a mapping
            if selector_key in self.reverse_mapping and display_value in self.reverse_mapping[selector_key]:
                actual_value = self.reverse_mapping[selector_key][display_value]
            else:
                # Find the actual value from the options list
                idx = selection[0]
                if idx < len(options):
                    actual_value = options[idx]
                else:
                    actual_value = display_value
            
            self.file_selector_vars[selector_key].set(actual_value)
            self.load_transcript_file()

    def setup_styles(self):
        """Sets up the text styles for different roles and the reasoning block."""
        self.conversation_text.tag_configure("system_style", background="#e0e0e0", 
                                           lmargin1=20, lmargin2=20)
        self.conversation_text.tag_configure("user_style", background="#d1e7dd", 
                                           lmargin1=20, lmargin2=20)
        self.conversation_text.tag_configure("assistant_style", background="#cce5ff", 
                                           lmargin1=20, lmargin2=20)
        self.conversation_text.tag_configure("unknown_style", background="#f8d7da", 
                                           lmargin1=20, lmargin2=20)
        self.conversation_text.tag_configure("role_header", font=("Helvetica", 11, "bold"))
        
        self.conversation_text.tag_configure("reasoning_header", font=("Helvetica", 10, "bold"))
        self.conversation_text.tag_configure("reasoning_style", font=("Helvetica", 10, "italic"), 
                                           lmargin1=30, lmargin2=30)
        self.conversation_text.tag_configure("main_text_header", font=("Helvetica", 10, "bold"))

    def bind_keys(self):
        self.bind("<Down>", self.next_conversation)
        self.bind("<Up>", self.previous_conversation)

    def load_transcript_file(self, event=None):
        """Load conversations from the selected experiment."""
        # Check if all file selectors have values
        selector_values = {}
        for selector in self.file_selectors:
            if isinstance(selector, tuple):
                selector_key = '_'.join(selector)
            else:
                selector_key = selector
            
            value = self.file_selector_vars[selector_key].get()
            if not value:
                return
            selector_values[selector_key] = value

        print(f"\n--- Loading data for: {selector_values} ---")

        self.all_conversations = []
        self.conversations = []
        self.current_conversation_index = -1

        try:
            # Find matching subfolders
            matching_folders = find_matching_subfolders(self.top_folder, 
                                                       self.file_selectors, 
                                                       selector_values)
            
            if not matching_folders:
                self.display_error("No matching experiment folders found")
                return
            
            if len(matching_folders) > 1:
                print(f"Warning: Found {len(matching_folders)} matching folders, using first one")
            
            folder = matching_folders[0]
            jsonl_path = os.path.join(folder, "results.jsonl")
            
            if not os.path.exists(jsonl_path):
                self.display_error(f"File not found: {jsonl_path}")
                return
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                self.all_conversations = [json.loads(line) for line in f]

            self.filter_conversations()

        except Exception as e:
            self.display_error(f"An error occurred: {e}")

    def filter_conversations(self):
        """Filter conversations based on row selectors."""
        if not self.all_conversations:
            return
        
        filtered = self.all_conversations.copy()
        
        # Apply each row selector filter
        for field_name, var in self.row_selector_vars.items():
            filter_value = var.get()
            if filter_value != "all":
                filtered = [conv for conv in filtered 
                           if conv.get(field_name) == filter_value]
        
        self.conversations = filtered
        
        if self.conversations:
            self.current_conversation_index = 0
            self.display_conversation()
        else:
            self.display_error("No conversations match the selected filters.")

    def display_conversation(self):
        """Displays a single conversation."""
        if not (0 <= self.current_conversation_index < len(self.conversations)):
            return
            
        self.conversation_text.delete(1.0, tk.END)

        data = self.conversations[self.current_conversation_index]
        
        if self.conversation_var_name in data:
            messages = data[self.conversation_var_name]
            thinking_blocks = data.get(self.thinking_blocks_var_name, []) if self.thinking_blocks_var_name else []
            assistant_message_index = 0

            for message in messages:
                role = message.get("role", "unknown")
                content = message.get("content", "No content available.")
                
                if role == 'assistant' and thinking_blocks:
                    reasoning_text = None
                    if assistant_message_index < len(thinking_blocks):
                        block = thinking_blocks[assistant_message_index]
                        if block is not None:
                            reasoning_text = block
                    
                    self.insert_message(role, content, reasoning=reasoning_text)
                    assistant_message_index += 1
                else:
                    self.insert_message(role, content)
        else:
            self.display_error(f"Error: The key '{self.conversation_var_name}' was not found in this record.")
        
        self.update_counter()

    def display_error(self, message):
        self.conversation_text.delete(1.0, tk.END)
        self.conversation_text.insert(tk.END, message, "unknown_style")
        self.counter_label.config(text="Error")

    def next_conversation(self, event=None):
        if self.current_conversation_index < len(self.conversations) - 1:
            self.current_conversation_index += 1
            self.display_conversation()

    def previous_conversation(self, event=None):
        if self.current_conversation_index > 0:
            self.current_conversation_index -= 1
            self.display_conversation()
            
    def insert_message(self, role: str, content: str, reasoning: str = None):
        """
        Inserts a message block into the text area.
        If reasoning is provided, it's formatted and displayed above the main content.
        """
        style_tag = f"{role.lower()}_style"
        self.conversation_text.insert(tk.END, f"{role.capitalize()}:\n", "role_header")

        if reasoning is not None:
            self.conversation_text.insert(tk.END, "Reasoning: ", ("reasoning_header", style_tag))
            self.conversation_text.insert(tk.END, f"{reasoning}\n", ("reasoning_style", style_tag))
            self.conversation_text.insert(tk.END, "\n", style_tag)
            self.conversation_text.insert(tk.END, "Main text: ", ("main_text_header", style_tag))
            self.conversation_text.insert(tk.END, f"{content}\n\n", style_tag)
        else:
            if role == 'assistant' and self.thinking_blocks_var_name:
                self.conversation_text.insert(tk.END, "Reasoning: ", ("reasoning_header", style_tag))
                self.conversation_text.insert(tk.END, "N/A\n", ("reasoning_style", style_tag))
                self.conversation_text.insert(tk.END, "\n", style_tag)
                self.conversation_text.insert(tk.END, "Main text: ", ("main_text_header", style_tag))

            self.conversation_text.insert(tk.END, f"{content}\n\n", style_tag)

    def update_counter(self):
        if self.conversations:
            total = len(self.conversations)
            current = self.current_conversation_index + 1
            self.counter_label.config(text=f"Conversation {current} of {total} (Use Up/Down arrows to navigate)")
        else:
            self.counter_label.config(text="No conversations loaded")


if __name__ == "__main__":
    app = TranscriptViewer(
        **email_assistant_vars
    )
    app.mainloop()