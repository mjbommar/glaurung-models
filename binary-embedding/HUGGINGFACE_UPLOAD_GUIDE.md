# HuggingFace Upload Guide

This guide explains how to upload the Glaurung Binary Embedding MICRO model to HuggingFace Hub.

## Prerequisites

### 1. Install Required Packages

```bash
pip install huggingface_hub transformers
```

### 2. Login to HuggingFace

You need to authenticate with HuggingFace before uploading. Choose one method:

**Method A: CLI Login (Recommended)**
```bash
huggingface-cli login
```

**Method B: Environment Variable**
```bash
export HF_TOKEN="your_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Verify Model Files

The upload script will verify these files exist:

```
models/micro-v6-best/
├── model.safetensors    (54 MB) - Model weights
├── config.json          (630 bytes) - Model configuration
├── tokenizer.json       (4.8 MB) - Tokenizer
├── README.md            (comprehensive documentation)
└── training_state.pt    (108 MB) - NOT uploaded (training only)
```

**Total upload size: ~59 MB**

## Quick Start

### Test Upload (Dry Run)

Verify everything is ready without actually uploading:

```bash
python upload_to_huggingface.py --dry-run
```

This will:
- Check if you're logged in ✓
- Verify all required files exist ✓
- Calculate upload size ✓
- Show what would be uploaded ✓

### Upload to Default Repository

```bash
python upload_to_huggingface.py
```

This uploads to: `mjbommar/glaurung-binary-embedding-micro`

### Upload to Custom Repository

```bash
python upload_to_huggingface.py --repo-name username/my-model-name
```

Replace `username` with your HuggingFace username.

### Create Private Repository

```bash
python upload_to_huggingface.py --private
```

You can make it public later from the repository settings.

## Upload Process

When you run the upload script:

1. **Login Check**: Verifies HuggingFace authentication
2. **File Verification**: Confirms all required files exist
3. **Size Calculation**: Shows total upload size (~59 MB)
4. **Confirmation Prompt**: Asks for confirmation before uploading
5. **Repository Creation**: Creates repo if it doesn't exist
6. **File Upload**: Uploads each file with progress indication
7. **Completion**: Provides link to your model

### Example Output

```
======================================================================
Glaurung Binary Embedding MICRO - HuggingFace Upload
======================================================================

Model directory: models/micro-v6-best
Repository: mjbommar/glaurung-binary-embedding-micro
Private: False
Dry run: False

✓ Logged in as: mjbommar

Verifying model files:
============================================================
✓ model.safetensors          ( 54.00 MB) - Model weights
✓ config.json                (  0.00 MB) - Model configuration
✓ tokenizer.json             (  4.80 MB) - Tokenizer configuration
✓ README.md                  (comprehensive documentation)
⚠️  training_state.pt (108.00 MB) - Will NOT be uploaded (training only)
============================================================
✓ All required files present

Total upload size: 58.80 MB

✓ Created .gitattributes for Git LFS

======================================================================
Ready to upload!
======================================================================
Repository: https://huggingface.co/mjbommar/glaurung-binary-embedding-micro
Size: 58.80 MB
Visibility: Public

Continue with upload? [y/N]: y

Creating new repository: mjbommar/glaurung-binary-embedding-micro
✓ Created repository

Uploading files:
============================================================
Uploading model.safetensors... ✓
Uploading config.json... ✓
Uploading tokenizer.json... ✓
Uploading README.md... ✓
============================================================
✓ All files uploaded successfully

======================================================================
✅ Upload Complete!
======================================================================

Your model is now available at:
🤗 https://huggingface.co/mjbommar/glaurung-binary-embedding-micro

To use the model:
  from transformers import AutoModelForMaskedLM, AutoTokenizer
  model = AutoModelForMaskedLM.from_pretrained("mjbommar/glaurung-binary-embedding-micro")
  tokenizer = AutoTokenizer.from_pretrained("mjbommar/glaurung-binary-embedding-micro")
```

## After Upload

### Verify Your Model

1. Visit your model page: `https://huggingface.co/username/repo-name`
2. Check that all files are present
3. Verify the README renders correctly
4. Test loading the model:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("username/repo-name")
tokenizer = AutoTokenizer.from_pretrained("username/repo-name")
print(f"Model loaded: {model.config.num_parameters:,} parameters")
```

### Share Your Model

Once uploaded, you can:

1. **Share the link**: `https://huggingface.co/username/repo-name`
2. **Add to collections**: Organize with other models
3. **Enable discussions**: Let users ask questions
4. **Add tags**: Help users discover your model
5. **Update README**: Add more examples and documentation

### Make Private Model Public

If you uploaded as private and want to make it public:

1. Go to: `https://huggingface.co/username/repo-name/settings`
2. Scroll to "Visibility"
3. Click "Make public"

## Troubleshooting

### "Not logged in to HuggingFace"

**Solution**: Run `huggingface-cli login` and enter your token

### "Repository already exists"

**Solution**: The script will update the existing repository. If you want a new repo, use `--repo-name` with a different name.

### "Permission denied"

**Solution**: Make sure you have write access to the repository. If it's under an organization, you need appropriate permissions.

### Upload is slow

**Solution**: The model is ~59 MB. Upload speed depends on your internet connection. Expected time:
- Fast (100 Mbps): ~5 minutes
- Medium (50 Mbps): ~10 minutes
- Slow (10 Mbps): ~45 minutes

### Files not showing on HuggingFace

**Solution**: Refresh the page. Large files may take a moment to process. Check the "Files and versions" tab.

## Advanced Options

### Upload from Different Directory

```bash
python upload_to_huggingface.py --model-dir path/to/your/model
```

### Script Help

```bash
python upload_to_huggingface.py --help
```

## What Gets Uploaded

### Included Files ✓

- `model.safetensors` - Model weights (54 MB)
- `config.json` - Model configuration (630 bytes)
- `tokenizer.json` - Tokenizer (4.8 MB)
- `README.md` - Documentation with model card
- `.gitattributes` - Git LFS configuration (auto-created)

### Excluded Files ✗

- `training_state.pt` - Training optimizer state (108 MB)
  - Only needed for resuming training
  - Not required for inference
  - Excluded to save bandwidth and storage

## Model Card Information

The uploaded README.md includes comprehensive documentation:

### Model Card Sections

1. **Model Description**: Overview and capabilities
2. **Intended Uses**: What the model is designed for
3. **Limitations**: Honest assessment of weaknesses
4. **Training Details**: Complete hyperparameters and procedure
5. **Evaluation Results**: Performance metrics
6. **Usage Examples**: Python code snippets
7. **Fine-Tuning Guide**: How to adapt for specific tasks
8. **Ethical Considerations**: Dual-use technology considerations
9. **Citation**: BibTeX entry for academic use

### Model Card Metadata

The README includes YAML frontmatter for HuggingFace indexing:

```yaml
language: code
license: apache-2.0
tags:
  - binary-analysis
  - malware-analysis
  - reverse-engineering
  - roberta
library_name: transformers
pipeline_tag: fill-mask
```

This helps users discover your model through HuggingFace search and filters.

## Post-Upload Checklist

- [ ] Verify model loads correctly from HuggingFace
- [ ] Check README renders properly
- [ ] Test inference with example code
- [ ] Add model to any relevant collections
- [ ] Share on social media / forums if desired
- [ ] Monitor discussions tab for user questions
- [ ] Consider writing a blog post about the model

## Next Steps

### Option 1: Use the Model

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("mjbommar/glaurung-binary-embedding-micro")
tokenizer = AutoTokenizer.from_pretrained("mjbommar/glaurung-binary-embedding-micro")

# Your analysis code here
```

### Option 2: Fine-Tune for Specific Task

```python
from transformers import AutoModelForSequenceClassification

# Start from the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "mjbommar/glaurung-binary-embedding-micro",
    num_labels=2  # e.g., malware classification
)

# Fine-tune on your labeled dataset
```

### Option 3: Contribute Improvements

- Report issues on the model's discussion page
- Share your fine-tuned variants
- Contribute to documentation
- Share interesting use cases

## Support

For questions about:
- **Model usage**: Check the README examples
- **Model issues**: Open discussion on HuggingFace
- **Upload problems**: Check troubleshooting section above
- **API questions**: See [HuggingFace Hub documentation](https://huggingface.co/docs/hub/index)

## License

The model is released under Apache 2.0 license, allowing broad use including commercial applications.
