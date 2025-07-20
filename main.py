import dspy
from dspy import Example
import os
import pandas as pd
import re
import sys
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

# ------------------------
# 1. Load story data from CSV FIRST
# ------------------------
print("📚 Loading story data from CSV...")
try:
    # Load the CSV file
    stories_df = pd.read_csv("story_content.csv")
    print(f"✅ Loaded {len(stories_df)} stories from CSV")
    print(f"📊 Columns: {', '.join(stories_df.columns.tolist())}")
    
    # Display first few stories for verification
    print("\n🔍 Sample stories:")
    for i, row in stories_df.head(3).iterrows():
        print(f"  • {row['title']}: {row['description'][:50]}...")
        
except FileNotFoundError:
    print("❌ story_content.csv not found! Please make sure the file is in the same directory.")
    print("🛑 Cannot proceed without story data. Exiting...")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    print("🛑 Cannot proceed without story data. Exiting...")
    sys.exit(1)

# ------------------------
# 2. Configure DSPy with OpenAI (REQUIRED for optimization)
# ------------------------
print("\n🔑 Checking for OpenAI API key...")

# Option 1: Set your OpenAI API key as an environment variable
# export OPENAI_API_KEY="your-api-key-here"

# Option 2: Set the API key directly (uncomment and add your key)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

if "OPENAI_API_KEY" not in os.environ:
    print("❌ OpenAI API key is REQUIRED for optimization!")
    print("\n🔧 To run optimization, you need to:")
    print("   1. Get an OpenAI API key from: https://platform.openai.com/api-keys")
    print("   2. Set it as environment variable: export OPENAI_API_KEY='your-key-here'")
    print("   3. Or uncomment line 36 above and add your key directly")
    print("\n💡 The optimization will compare performance before/after MIPROv2 tuning")
    sys.exit(1)
else:
    print("✅ OpenAI API key found! Proceeding with optimization...")
    dspy.configure(lm=dspy.LM(model="openai/gpt-4o"))

# ------------------------
# 3. Define intelligent search function with fuzzy matching
# ------------------------
def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def search_items(query: str = "", tags: list = None) -> list[str]:
    """
    Search through stories based on query and/or tags with fuzzy matching.
    Returns a list of story titles that match the criteria.
    
    Args:
        query: Text query to search for
        tags: List of tags to match against
    """
    matching_stories = []
    
    # Handle both query and tags parameters
    search_terms = []
    if query:
        search_terms.extend(re.findall(r'\b\w+\b', query.lower()))
    if tags:
        search_terms.extend([tag.lower().strip() for tag in tags])
    
    if not search_terms:
        return []
    
    for _, story in stories_df.iterrows():
        score = 0
        title = story['title'].lower()
        description = story['description'].lower()
        story_tags = [tag.strip().lower() for tag in story['tags'].split(',')]
        
        # Exact matching (higher scores)
        for term in search_terms:
            if term in title:
                score += 5
            if term in description:
                score += 3
            if term in story['tags'].lower():
                score += 4
        
        # Fuzzy matching for tags (moderate scores)
        for search_tag in search_terms:
            for story_tag in story_tags:
                sim = similarity(search_tag, story_tag)
                if sim >= 0.8:  # High similarity threshold
                    score += 3
                elif sim >= 0.6:  # Medium similarity threshold
                    score += 2
                elif sim >= 0.4:  # Low similarity threshold
                    score += 1
        
        # Fuzzy matching for descriptions (lower scores)
        for term in search_terms:
            # Check for partial matches in description
            desc_words = description.split()
            for word in desc_words:
                if similarity(term, word) >= 0.7:
                    score += 1
        
        # Add story if it has any matches
        if score > 0:
            matching_stories.append((story['title'], score))
    
    # Sort by score and return top matches
    matching_stories.sort(key=lambda x: x[1], reverse=True)
    return [story[0] for story in matching_stories[:5]]  # Return top 5 matches

# ------------------------
# 4. Create training dataset dynamically from CSV data
# ------------------------
def create_training_data():
    """Create comprehensive training examples using actual stories from the CSV"""
    
    # Find stories with specific characteristics from the CSV
    sci_fi_emotional = stories_df[
        (stories_df['tags'].str.contains('sci-fi', case=False)) & 
        (stories_df['tags'].str.contains('emotional', case=False))
    ]['title'].tolist()
    
    philosophical = stories_df[
        stories_df['tags'].str.contains('philosophical', case=False)
    ]['title'].tolist()
    
    fantasy_emotional = stories_df[
        (stories_df['tags'].str.contains('fantasy', case=False)) & 
        (stories_df['tags'].str.contains('emotional|introspective', case=False))
    ]['title'].tolist()
    
    psychological = stories_df[
        stories_df['tags'].str.contains('psychological', case=False)
    ]['title'].tolist()
    
    ai_stories = stories_df[
        stories_df['tags'].str.contains('AI|artificial-intelligence', case=False)
    ]['title'].tolist()
    
    memory_stories = stories_df[
        stories_df['tags'].str.contains('memory', case=False)
    ]['title'].tolist()
    
    coming_of_age = stories_df[
        stories_df['tags'].str.contains('coming-of-age', case=False)
    ]['title'].tolist()
    
    dystopian = stories_df[
        stories_df['tags'].str.contains('dystopian', case=False)
    ]['title'].tolist()
    
    magical_realism = stories_df[
        stories_df['tags'].str.contains('magical-realism', case=False)
    ]['title'].tolist()
    
    literary_fiction = stories_df[
        stories_df['tags'].str.contains('literary', case=False)
    ]['title'].tolist()
    
    contemplative = stories_df[
        stories_df['tags'].str.contains('contemplative|introspective', case=False)
    ]['title'].tolist()
    
    consciousness_themes = stories_df[
        stories_df['tags'].str.contains('consciousness|identity', case=False)
    ]['title'].tolist()
    
    # Create comprehensive training examples
    trainset = []
    
    # Core preference examples
    if sci_fi_emotional:
        trainset.append(
            Example(
                user_query="I love emotional sci-fi stories", 
                tags=["sci-fi", "emotional", "introspective", "character-development"],
                recommendations=sci_fi_emotional[:2]
            ).with_inputs("user_query")
        )
        
        trainset.append(
            Example(
                user_query="Give me sci-fi that makes me feel something deep", 
                tags=["sci-fi", "emotional", "deep", "moving", "character-driven"],
                recommendations=sci_fi_emotional[:2]
            ).with_inputs("user_query")
        )
    
    if philosophical:
        trainset.append(
            Example(
                user_query="I'm into introspective philosophical tales", 
                tags=["philosophical", "introspective", "contemplative", "literary"],
                recommendations=philosophical[:2]
            ).with_inputs("user_query")
        )
        
        trainset.append(
            Example(
                user_query="Stories that explore big questions about existence", 
                tags=["philosophical", "existential", "deep-thinking", "contemplative"],
                recommendations=philosophical[:2]
            ).with_inputs("user_query")
        )
    
    if fantasy_emotional:
        trainset.append(
            Example(
                user_query="Looking for fantasy with emotional depth", 
                tags=["fantasy", "emotional", "character-driven", "depth"],
                recommendations=fantasy_emotional[:2]
            ).with_inputs("user_query")
        )
    
    if psychological:
        trainset.append(
            Example(
                user_query="I want psychological stories that make me think", 
                tags=["psychological", "thought-provoking", "introspective", "complex"],
                recommendations=psychological[:2]
            ).with_inputs("user_query")
        )
        
        trainset.append(
            Example(
                user_query="Complex psychological narratives about the human mind", 
                tags=["psychological", "complex", "human-nature", "mind", "introspective"],
                recommendations=psychological[:2]
            ).with_inputs("user_query")
        )
    
    if ai_stories:
        trainset.append(
            Example(
                user_query="Stories about AI and consciousness", 
                tags=["AI", "consciousness", "sci-fi", "philosophical", "technology"],
                recommendations=ai_stories[:2]
            ).with_inputs("user_query")
        )
        
        trainset.append(
            Example(
                user_query="What happens when machines start to think and feel?", 
                tags=["AI", "artificial-intelligence", "emotions", "consciousness", "technology"],
                recommendations=ai_stories[:2]
            ).with_inputs("user_query")
        )
    
    # Memory and identity themes
    if memory_stories:
        trainset.append(
            Example(
                user_query="Stories exploring memory and how it shapes us", 
                tags=["memory", "identity", "psychological", "introspective", "philosophical"],
                recommendations=memory_stories[:2]
            ).with_inputs("user_query")
        )
    
    # Coming of age with depth
    if coming_of_age:
        trainset.append(
            Example(
                user_query="Coming-of-age stories with psychological complexity", 
                tags=["coming-of-age", "psychological", "character-development", "complex", "young-adult"],
                recommendations=coming_of_age[:2]
            ).with_inputs("user_query")
        )
    
    # Dystopian with thoughtfulness
    if dystopian:
        trainset.append(
            Example(
                user_query="Thoughtful dystopian stories that aren't just action", 
                tags=["dystopian", "thoughtful", "literary", "social-commentary", "introspective"],
                recommendations=dystopian[:2]
            ).with_inputs("user_query")
        )
    
    # Literary and contemplative
    if literary_fiction:
        trainset.append(
            Example(
                user_query="Literary stories that blur genre boundaries", 
                tags=["literary", "genre-blending", "contemplative", "sophisticated", "artistic"],
                recommendations=literary_fiction[:2]
            ).with_inputs("user_query")
        )
    
    # Consciousness exploration
    if consciousness_themes:
        trainset.append(
            Example(
                user_query="What does it mean to be conscious and aware?", 
                tags=["consciousness", "identity", "philosophical", "existential", "self-awareness"],
                recommendations=consciousness_themes[:2]
            ).with_inputs("user_query")
        )
    
    # Broader thematic examples using search results
    trainset.append(
        Example(
            user_query="Slow-burn stories with deep characters", 
            tags=["slow-burn", "character-development", "literary", "contemplative"],
            recommendations=search_items("slow burn character development")[:2]
        ).with_inputs("user_query")
    )
    
    trainset.append(
        Example(
            user_query="Stories about human connection and relationships", 
            tags=["relationships", "human-connection", "emotional", "character-driven", "empathy"],
            recommendations=search_items("", ["relationships", "emotional", "connection"])[:2]
        ).with_inputs("user_query")
    )
    
    trainset.append(
        Example(
            user_query="Melancholic stories that stay with you", 
            tags=["melancholic", "bittersweet", "emotional", "memorable", "contemplative"],
            recommendations=search_items("", ["melancholic", "bittersweet", "emotional"])[:2]
        ).with_inputs("user_query")
    )
    
    trainset.append(
        Example(
            user_query="Stories that blend reality with something magical", 
            tags=["magical-realism", "literary", "blend", "reality", "fantasy"],
            recommendations=search_items("", ["magical-realism", "literary", "fantasy"])[:2]
        ).with_inputs("user_query")
    )
    
    trainset.append(
        Example(
            user_query="I want stories about transformation and growth", 
            tags=["transformation", "growth", "character-development", "personal-journey", "introspective"],
            recommendations=search_items("", ["transformation", "growth", "character-development"])[:2]
        ).with_inputs("user_query")
    )
    
    return trainset

# ------------------------
# 5. Create evaluation dataset (separate from training)
# ------------------------
def create_evaluation_data():
    """Create comprehensive test examples for evaluation - PROPERLY FORMATTED"""
    
    eval_set = [
        Example(
            user_query="I'm looking for a sci-fi story with strong emotions", 
            tags=["sci-fi", "emotional", "strong-emotions", "character-driven"],
            recommendations=search_items("sci-fi emotional")[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Something introspective and philosophical", 
            tags=["introspective", "philosophical", "contemplative", "literary"],
            recommendations=search_items("introspective philosophical")[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="I want stories about AI and consciousness", 
            tags=["AI", "consciousness", "artificial-intelligence", "philosophical", "technology"],
            recommendations=search_items("AI consciousness")[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Fantasy with emotional depth and character development", 
            tags=["fantasy", "emotional", "character-development", "depth", "literary"],
            recommendations=search_items("fantasy emotional character")[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Slow-burn stories with psychological themes", 
            tags=["slow-burn", "psychological", "contemplative", "introspective"],
            recommendations=search_items("slow psychological")[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Stories that explore memory and identity", 
            tags=["memory", "identity", "psychological", "introspective", "philosophical"],
            recommendations=search_items("memory identity")[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Thoughtful stories with philosophical depth", 
            tags=["thoughtful", "philosophical", "contemplative", "literary", "depth"],
            recommendations=search_items("thoughtful philosophical")[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Emotional stories about relationships and connection", 
            tags=["emotional", "relationships", "connection", "character-driven", "human-connection"],
            recommendations=search_items("emotional relationship")[:2]
        ).with_inputs("user_query"),
        
        # Additional evaluation examples for broader coverage
        Example(
            user_query="Stories about consciousness and what makes us human", 
            tags=["consciousness", "humanity", "philosophical", "identity", "existential"],
            recommendations=search_items("", ["consciousness", "identity", "philosophical"])[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Melancholic stories with beautiful prose", 
            tags=["melancholic", "literary", "beautiful-prose", "contemplative", "bittersweet"],
            recommendations=search_items("", ["melancholic", "literary", "contemplative"])[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Coming-of-age with psychological complexity", 
            tags=["coming-of-age", "psychological", "complex", "character-development", "introspective"],
            recommendations=search_items("", ["coming-of-age", "psychological", "complex"])[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Stories that blend genres in thoughtful ways", 
            tags=["genre-blending", "literary", "thoughtful", "innovative", "artistic"],
            recommendations=search_items("", ["literary", "genre-blending", "contemplative"])[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="What happens when technology changes how we think?", 
            tags=["technology", "consciousness", "change", "philosophical", "sci-fi"],
            recommendations=search_items("", ["technology", "consciousness", "sci-fi"])[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Stories about transformation and personal growth", 
            tags=["transformation", "personal-growth", "character-development", "introspective", "journey"],
            recommendations=search_items("", ["transformation", "growth", "character-development"])[:2]
        ).with_inputs("user_query"),
        
        Example(
            user_query="Quiet stories that make you think deeply", 
            tags=["quiet", "contemplative", "thought-provoking", "introspective", "literary"],
            recommendations=search_items("", ["contemplative", "introspective", "literary"])[:2]
        ).with_inputs("user_query")
    ]
    
    return eval_set

# ------------------------
# 6. Create a ReAct agent with custom system prompt
# ------------------------
try:
    instructions = f"""You are an expert story recommendation assistant with access to a curated collection of {len(stories_df)} stories.

INITIAL TASTE PROFILE (use this as your baseline preference model):
You have a strong preference for:
- Emotional depth and character development over plot-driven action

You tend to avoid: fast-paced action, pure horror, simple adventure stories without depth.

YOUR TASK:
1. Analyze the user's query to understand their preferences
2. Generate a list of relevant tags that capture their interests (extract/infer from their query)
3. Use the search_items tool with both the original query AND the generated tags to find matching stories
4. Return story titles that align with both the user's stated preferences AND your taste profile

ALWAYS generate tags first, then search using both query and tags for better matching."""
    
    signature = dspy.Signature("user_query -> tags, recommendations", instructions)
    recommender = dspy.ReAct(signature, tools=[search_items])
    
    # Create training and evaluation data from CSV
    trainset = create_training_data()
    evalset = create_evaluation_data()
    
    print(f"\n📚 Created {len(trainset)} training examples from CSV data")
    for i, example in enumerate(trainset, 1):
        print(f"   {i}. '{example.user_query}'")
        print(f"      Tags: {example.tags}")
        print(f"      Recommendations: {example.recommendations}")
    
    print(f"\n🎯 Created {len(evalset)} evaluation examples")
    
    # ------------------------
    # 7. Define comprehensive evaluation metrics
    # ------------------------
    def exact_match_metric(example, pred, trace=None):
        """Check if any predicted recommendations exactly match expected ones"""
        if not hasattr(pred, "recommendations"):
            return False
        expected = [rec.lower().strip() for rec in example.recommendations]
        actual = [rec.lower().strip() for rec in pred.recommendations] if isinstance(pred.recommendations, list) else [pred.recommendations.lower().strip()]
        return any(exp in actual for exp in expected)
    
    def partial_overlap_metric(example, pred, trace=None):
        """More lenient - checks if there's any overlap in recommendations"""
        if not hasattr(pred, "recommendations"):
            return False
        expected = set([rec.lower().strip() for rec in example.recommendations])
        actual = set([rec.lower().strip() for rec in pred.recommendations] if isinstance(pred.recommendations, list) else [pred.recommendations.lower().strip()])
        return len(expected.intersection(actual)) > 0
    
    def recommendation_relevance_metric(example, pred, trace=None):
        """Check if predicted recommendations are relevant using both query and generated tags"""
        if not hasattr(pred, "recommendations"):
            return False
        
        # Use both query and predicted tags for better relevance checking
        query = example.user_query
        pred_tags = getattr(pred, "tags", []) if hasattr(pred, "tags") else []
        
        # Get relevant stories using the enhanced search
        relevant_stories = set([story.lower().strip() for story in search_items(query, pred_tags)])
        actual = set([rec.lower().strip() for rec in pred.recommendations] if isinstance(pred.recommendations, list) else [pred.recommendations.lower().strip()])
        
        # Return True if any actual recommendations are in the relevant set
        return len(relevant_stories.intersection(actual)) > 0
    
    def tag_quality_metric(example, pred, trace=None):
        """Evaluate the quality of generated tags by checking overlap with expected tags"""
        if not hasattr(pred, "tags") or not hasattr(example, "tags"):
            return False
        
        expected_tags = set([tag.lower().strip() for tag in example.tags])
        actual_tags = set([tag.lower().strip() for tag in pred.tags] if isinstance(pred.tags, list) else [pred.tags.lower().strip()])
        
        # Check for exact matches and fuzzy matches
        exact_matches = len(expected_tags.intersection(actual_tags))
        
        # Check for fuzzy matches
        fuzzy_matches = 0
        for expected_tag in expected_tags:
            for actual_tag in actual_tags:
                if similarity(expected_tag, actual_tag) >= 0.7:
                    fuzzy_matches += 1
                    break
        
        # Return True if we have at least 2 good matches (exact or fuzzy)
        return (exact_matches + fuzzy_matches) >= 2
    
    # ------------------------
    # 8. Evaluate baseline (before optimization)
    # ------------------------
    print("\n" + "="*60)
    print("📊 BASELINE EVALUATION (Before Optimization)")
    print("="*60)
    
    def evaluate_model(model, dataset, metric_name="exact_match"):
        """Evaluate a model on a dataset"""
        if metric_name == "exact_match":
            metric = exact_match_metric
        elif metric_name == "partial_overlap":
            metric = partial_overlap_metric
        elif metric_name == "tag_quality":
            metric = tag_quality_metric
        else:
            metric = recommendation_relevance_metric
            
        correct = 0
        total = len(dataset)
        
        for example in dataset:
            try:
                pred = model(user_query=example.user_query)
                if metric(example, pred):
                    correct += 1
                    print(f"✅ Query: '{example.user_query[:40]}...'")
                    if metric_name == "tag_quality":
                        print(f"   Expected tags: {example.tags}")
                        print(f"   Generated tags: {getattr(pred, 'tags', 'None')}")
                    else:
                        print(f"   Expected: {example.recommendations}")
                        print(f"   Got: {getattr(pred, 'recommendations', 'None')}")
                        print(f"   Tags: {getattr(pred, 'tags', 'None')}")
                else:
                    print(f"❌ Query: '{example.user_query[:40]}...'")
                    if metric_name == "tag_quality":
                        print(f"   Expected tags: {example.tags}")
                        print(f"   Generated tags: {getattr(pred, 'tags', 'None')}")
                    else:
                        print(f"   Expected: {example.recommendations}")
                        print(f"   Got: {getattr(pred, 'recommendations', 'None')}")
                        print(f"   Tags: {getattr(pred, 'tags', 'None')}")
            except Exception as e:
                print(f"⚠️  Error on query '{example.user_query[:40]}...': {e}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📈 {metric_name.upper()} ACCURACY: {correct}/{total} = {accuracy:.2%}")
        return accuracy
    
    # Run baseline evaluation
    baseline_exact = evaluate_model(recommender, evalset, "exact_match")
    print(f"\n🔍 Running additional metrics...")
    baseline_overlap = evaluate_model(recommender, evalset, "partial_overlap")
    baseline_relevance = evaluate_model(recommender, evalset, "relevance")
    baseline_tags = evaluate_model(recommender, evalset, "tag_quality")
    
    # ------------------------
    # 9. Run optimization with MIPROv2
    # ------------------------
    print("\n" + "="*60)
    print("🔧 RUNNING OPTIMIZATION (MIPROv2)")
    print("="*60)
    
    def composite_metric(example, pred, trace=None):
        """Composite metric that evaluates both recommendations and tag quality"""
        rec_score = 1 if partial_overlap_metric(example, pred, trace) else 0
        tag_score = 1 if tag_quality_metric(example, pred, trace) else 0
        # Both need to be good for the example to pass
        return (rec_score + tag_score) >= 1  # At least one should be good
    
    tp = dspy.MIPROv2(
        metric=composite_metric, 
        auto="medium",  # light/medium/heavy - increased for better optimization
        num_threads=6,  # Increased for better parallelization
        max_errors=10   # Increased for larger training set
    )
    
    print("🚀 Starting MIPROv2 optimization...")
    print("⏳ This may take a few minutes...")
    
    try:
        optimized_recommender = tp.compile(
            recommender, 
            trainset=trainset, 
            requires_permission_to_run=False,
            max_bootstrapped_demos=6,  # Increased for larger training set
            max_labeled_demos=4        # Increased for better optimization
        )
        print("✅ Optimization complete!")
    except Exception as e:
        print(f"❌ Error during MIPROv2 optimization: {e}")
        print("🔄 Falling back to unoptimized model for comparison...")
        optimized_recommender = recommender
    
    # ------------------------
    # 10. Evaluate optimized model
    # ------------------------
    print("\n" + "="*60)
    print("📊 OPTIMIZED EVALUATION (After Optimization)")
    print("="*60)
    
    optimized_exact = evaluate_model(optimized_recommender, evalset, "exact_match")
    print(f"\n🔍 Running additional metrics...")
    optimized_overlap = evaluate_model(optimized_recommender, evalset, "partial_overlap")
    optimized_relevance = evaluate_model(optimized_recommender, evalset, "relevance")
    optimized_tags = evaluate_model(optimized_recommender, evalset, "tag_quality")
    
    # ------------------------
    # 11. Show improvement comparison
    # ------------------------
    print("\n" + "="*60)
    print("📈 OPTIMIZATION RESULTS COMPARISON")
    print("="*60)
    
    def show_improvement(metric_name, baseline, optimized):
        improvement = optimized - baseline
        improvement_pct = (improvement / baseline * 100) if baseline > 0 else 0
        
        print(f"\n📊 {metric_name.upper()}:")
        print(f"   Before: {baseline:.2%}")
        print(f"   After:  {optimized:.2%}")
        print(f"   Change: {improvement:+.2%} ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print(f"   🎉 IMPROVED by {improvement:.2%}!")
        elif improvement < 0:
            print(f"   📉 Decreased by {abs(improvement):.2%}")
        else:
            print(f"   📊 No change")
    
    show_improvement("Exact Match Accuracy", baseline_exact, optimized_exact)
    show_improvement("Partial Overlap Accuracy", baseline_overlap, optimized_overlap)
    show_improvement("Relevance Accuracy", baseline_relevance, optimized_relevance)
    show_improvement("Tag Quality Accuracy", baseline_tags, optimized_tags)
    
    # Overall summary
    total_baseline = (baseline_exact + baseline_overlap + baseline_relevance + baseline_tags) / 4
    total_optimized = (optimized_exact + optimized_overlap + optimized_relevance + optimized_tags) / 4
    
    print(f"\n🏆 OVERALL AVERAGE:")
    print(f"   Before: {total_baseline:.2%}")
    print(f"   After:  {total_optimized:.2%}")
    print(f"   Improvement: {total_optimized - total_baseline:+.2%}")
    
    if total_optimized > total_baseline:
        print(f"\n🎉 SUCCESS! MIPROv2 optimization improved performance!")
    else:
        print(f"\n📝 Note: Small dataset may limit optimization gains. Try with more training examples.")
    
    print(f"\n✨ All recommendations are from the {len(stories_df)} stories in story_content.csv")
    
except Exception as e:
    print(f"❌ Error running the script: {e}")
    print("\n🔧 Quick fixes:")
    print("1. Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key-here'")
    print("2. Make sure story_content.csv is in the same directory")
    print("3. Install required packages: pip install -r requirements.txt")
