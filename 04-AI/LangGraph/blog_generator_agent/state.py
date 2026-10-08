from pydantic import BaseModel
import warnings 
warnings.filterwarnings('ignore')

class BlogState(BaseModel):
    # User Input field
    topic : str = ""
    audience : str = "General reader"
    # researcher Output field
    research : str = ""
    research_feedback : str = ""
    # Writer Output filed
    draft : str = ""
    draft_feedback : str = ""
    # Editor Output filed
    final_blog : str = ""
    # Meta Data Field
    revision_count : int = 0


