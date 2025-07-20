# 📚 Story Recommendation System with DSPy Optimization

A sophisticated story recommendation system that uses **DSPy's MIPROv2** to optimize recommendation quality through automated prompt engineering and fuzzy matching algorithms.

## 🎯 **What This System Does**

This system creates an AI-powered story recommendation agent that:
- **Analyzes user queries** to understand preferences
- **Generates relevant tags** from user requests
- **Searches through stories** using fuzzy matching
- **Recommends stories** that match both user preferences and a curated taste profile
- **Optimizes its performance** using machine learning techniques

## 📋 **Prerequisites**

### **Required API Keys**
- **OpenAI API Key**: Required for MIPROv2 optimization
  - Get one from: https://platform.openai.com/api-keys
  - Set as environment variable: `export OPENAI_API_KEY='your-key-here'`

### **Required Files**
- `story_content.csv` - Your story database (included)
- `requirements.txt` - Python dependencies
- `test2.py` - Main optimization script

### **Python Packages**
```bash
pip install dspy-ai pandas python-dotenv
```

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Clone or download the project
# Navigate to project directory
cd story-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-openai-api-key-here'
```

### **2. Run the Optimization**
```bash
python main.py
```

## 📊 **Understanding the Output**

### **Phase 1: Data Loading**
```
📚 Loading story data from CSV...
✅ Loaded 30 stories from CSV
📊 Columns: title, description, tags
```
- Confirms your story database loaded successfully
- Shows how many stories are available

### **Phase 2: Training Data Creation**
```
📚 Created 15 training examples from CSV data
   1. 'I love emotional sci-fi stories'
      Tags: ['sci-fi', 'emotional', 'introspective', 'character-development']
      Recommendations: ['Refractions', 'The Long Silence']
```
- Shows the training examples created from your story data
- Each example maps user queries to expected tags and story recommendations

### **Phase 3: Baseline Evaluation (Before Optimization)**
```
============================================================
📊 BASELINE EVALUATION (Before Optimization)
============================================================

✅ Query: 'I'm looking for a sci-fi story with strong...'
   Expected: ['Refractions', 'The Long Silence']
   Got: ['Steel and Rain', 'Refractions']
   Tags: ['sci-fi', 'emotional', 'technology']

📈 EXACT MATCH ACCURACY: 2/15 = 13.33%
📈 PARTIAL OVERLAP ACCURACY: 8/15 = 53.33%
📈 RELEVANCE ACCURACY: 12/15 = 80.00%
📈 TAG QUALITY ACCURACY: 7/15 = 46.67%
```

**What These Metrics Mean:**
- **Exact Match**: Predicted recommendations exactly match expected ones
- **Partial Overlap**: At least one predicted recommendation matches expected
- **Relevance**: Predicted recommendations are relevant to the query (using fuzzy search)
- **Tag Quality**: Generated tags match expected tags (exact + fuzzy matching)

### **Phase 4: MIPROv2 Optimization**
```
============================================================
🔧 RUNNING OPTIMIZATION (MIPROv2)
============================================================
🚀 Starting MIPROv2 optimization...
⏳ This may take a few minutes...
✅ Optimization complete!
```
- MIPROv2 analyzes training examples and optimizes the system prompt
- Uses medium-intensity optimization with 6 parallel threads
- May take 5-15 minutes depending on your system

### **Phase 5: Optimized Evaluation (After Optimization)**
```
============================================================
📊 OPTIMIZED EVALUATION (After Optimization)
============================================================

✅ Query: 'I'm looking for a sci-fi story with strong...'
   Expected: ['Refractions', 'The Long Silence']
   Got: ['Refractions', 'The Long Silence']
   Tags: ['sci-fi', 'emotional', 'character-driven', 'introspective']

📈 EXACT MATCH ACCURACY: 6/15 = 40.00%
📈 PARTIAL OVERLAP ACCURACY: 13/15 = 86.67%
📈 RELEVANCE ACCURACY: 14/15 = 93.33%
📈 TAG QUALITY ACCURACY: 11/15 = 73.33%
```

### **Phase 6: Results Comparison**
```
============================================================
📈 OPTIMIZATION RESULTS COMPARISON
============================================================

📊 EXACT MATCH ACCURACY:
   Before: 13.33%
   After:  40.00%
   Change: +26.67% (+200.0%)
   🎉 IMPROVED by 26.67%!

📊 PARTIAL OVERLAP ACCURACY:
   Before: 53.33%
   After:  86.67%
   Change: +33.33% (+62.5%)
   🎉 IMPROVED by 33.33%!

🏆 OVERALL AVERAGE:
   Before: 48.33%
   After:  73.33%
   Improvement: +25.00%

🎉 SUCCESS! MIPROv2 optimization improved performance!
```

## 🔧 **System Architecture**

### **Core Components**

1. **Story Database** (`story_content.csv`)
   - 30 curated stories with titles, descriptions, and comprehensive tags
   - Each story has 10+ tags covering genre, theme, tone, complexity

2. **Fuzzy Search Engine** (`search_items()`)
   - Exact matching (highest scores)
   - Fuzzy tag matching using similarity algorithms
   - Multi-criteria scoring system

3. **ReAct Agent** (DSPy)
   - System prompt with embedded taste profile
   - Tool access to search function
   - Generates tags and recommendations

4. **Optimization Engine** (MIPROv2)
   - Analyzes training examples
   - Optimizes system prompt automatically
   - Uses composite metric for evaluation

### **Data Flow**
```
User Query → Tag Generation → Fuzzy Search → Story Recommendations
     ↑              ↑              ↑              ↑
  System Prompt  Taste Profile  Story Database  Final Selection
```

## 📈 **Evaluation Metrics Explained**

### **1. Exact Match Accuracy**
- **What it measures**: Perfect matches between predicted and expected recommendations
- **Good score**: >30%
- **Perfect score**: 100% (rarely achieved)

### **2. Partial Overlap Accuracy**
- **What it measures**: Any overlap between predicted and expected recommendations
- **Good score**: >70%
- **Perfect score**: 100%

### **3. Relevance Accuracy**
- **What it measures**: Whether predicted recommendations are relevant to the query
- **Good score**: >80%
- **Perfect score**: 100%

### **4. Tag Quality Accuracy**
- **What it measures**: How well generated tags match expected tags
- **Good score**: >60%
- **Perfect score**: 100%

## ⚙️ **Configuration Options**

### **Optimization Intensity**
```python
tp = dspy.MIPROv2(
    auto="medium",  # Options: "light", "medium", "heavy"
    num_threads=6,  # Parallel processing threads
    max_errors=10   # Error tolerance
)
```

### **Training Parameters**
```python
max_bootstrapped_demos=6,  # Auto-generated examples
max_labeled_demos=4        # Human-labeled examples
```

### **Search Parameters**
```python
# Fuzzy matching thresholds
if sim >= 0.8:    # High similarity
    score += 3
elif sim >= 0.6:  # Medium similarity
    score += 2
elif sim >= 0.4:  # Low similarity
    score += 1
```


### **Adding Training Examples**
In `create_training_data()`, add new examples:
```python
trainset.append(
    Example(
        user_query="Your custom query here",
        tags=["tag1", "tag2", "tag3"],
        recommendations=search_items("your search terms")[:2]
    ).with_inputs("user_query")
)
```

### **Performance Tips**

- **Small dataset**: Use `auto="light"` for faster optimization
- **Large dataset**: Use `auto="heavy"` for better results
- **Memory issues**: Reduce `num_threads` and `max_bootstrapped_demos`
- **API limits**: Add delays between optimization runs
