import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, font
from PIL import Image, ImageTk
import base64
import os
import asyncio
import aiohttp
from urllib.parse import quote
from bs4 import BeautifulSoup
import logging
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import markdown
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResponseCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.usage_queue = deque()
        
    def get(self, key):
        if key in self.cache:
            self.usage_queue.remove(key)
            self.usage_queue.append(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            self.usage_queue.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.usage_queue.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.usage_queue.append(key)
        
response_cache = ResponseCache()

async def search_duckduckgo_async(query, max_results=5):
    try:
        from duckduckgo_search import DDGS
        search = DDGS()
        results = list(search.text(query, max_results=max_results))
        return [{'title': r['title'], 'link': r['href'], 'snippet': r['body']} for r in results]
    except Exception as e:
        logging.error(f"An error occurred in DuckDuckGo search: {e}")
        return None
    
async def search_wikipedia_async(query, max_results=5):
    try:
        import wikipedia
        wikipedia.set_lang("en")
        search_results = wikipedia.search(query, results=max_results)
        summaries = []
        for title in search_results:
            try:
                page = wikipedia.page(title)
                summaries.append({'title': title, 'link': page.url, 'snippet': page.summary[:200]})
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                continue
        return summaries
    except Exception as e:
        logging.error(f"An error occurred in Wikipedia search: {e}")
        return None
    
async def search_yahoo_async(query):
    url = f'https://search.yahoo.com/search?p={quote(query)}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                results = soup.find_all('h3', {'class': 'title'})
                return [{'title': r.get_text(strip=True), 'link': r.find('a').get('href', ''), 'snippet': ''} for r in results if r.find('a')]
            else:
                logging.error(f"Error in Yahoo search: {response.status} - {await response.text()}")
                return None
            
class AdvancedChatUI:
    def __init__(self, master):
        self.master = master
        master.title("Mix of Agents MurMur")
        master.geometry("1000x800")
        
        self.setup_variables()
        self.setup_ui()
        
    def setup_ui(self):
        self.setup_font_options()
        self.setup_chat_frame()
        self.setup_input_frame()
        self.setup_preference_ui()
        
    def setup_variables(self):
        self.font_size = tk.IntVar(value=15)
        self.current_image_path = None
        self.current_image_base64 = None
        self.image_counter = 0
        self.chat_history = []
        self.user_preferences = {
            "style": tk.StringVar(value="neutral"),
            "detail_level": tk.StringVar(value="medium"),
            "preferred_sources": tk.StringVar(value="all")
        }
        self.models = {
            "layer1": ["mistral", "llava:13b"],
            "layer2": ["llama3:8b", "qwen2:7b"],
            "final": "gemma2:9b"
        }
        self.use_duckduckgo = tk.BooleanVar(value=False)
        self.use_wikipedia = tk.BooleanVar(value=False)
        self.use_yahoo = tk.BooleanVar(value=False)
        
    def setup_font_options(self):
        font_options_frame = ttk.Frame(self.master, padding="5")
        font_options_frame.pack(fill=tk.X)
        
        ttk.Label(font_options_frame, text="Font Size:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(font_options_frame, from_=8, to=32, textvariable=self.font_size, width=5, command=self.update_font_size).pack(side=tk.LEFT, padx=5)
        
        self.timer_label = ttk.Label(font_options_frame, text="Time: 0.0s")
        self.timer_label.pack(side=tk.LEFT, padx=10)
        
    def setup_chat_frame(self):
        chat_frame = ttk.Frame(self.master, padding="10")
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, width=80, height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.config(foreground="#A0A0A0")
        self.chat_display.tag_configure('orange', foreground='#FF8C00')
        
    def setup_input_frame(self):
        input_frame = ttk.Frame(self.master, padding="10")
        input_frame.pack(fill=tk.X)
        
        self.user_input = ttk.Entry(input_frame, width=70)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)
        
        ttk.Button(input_frame, text="Send", command=self.send_message).pack(side=tk.LEFT)
        ttk.Button(input_frame, text="Upload File", command=self.upload_image).pack(side=tk.LEFT)
        ttk.Button(input_frame, text="Reset", command=self.reset_chat).pack(side=tk.LEFT)
        
    def setup_preference_ui(self):
        pref_frame = ttk.Frame(self.master, padding="10")
        pref_frame.pack(fill=tk.X, padx=10, pady=5)
        
        preferences = [
            ("Writing Style:", "style", ["formal", "casual", "neutral"]),
            ("Detail Level:", "detail_level", ["low", "medium", "high"]),
            ("Preferred Sources:", "preferred_sources", ["all", "academic", "news", "wiki"])
        ]
        
        for i, (label, key, values) in enumerate(preferences):
            ttk.Label(pref_frame, text=label).grid(row=i, column=0, sticky="w")
            ttk.Combobox(pref_frame, textvariable=self.user_preferences[key], values=values).grid(row=i, column=1, sticky="w")
            
        ttk.Checkbutton(pref_frame, text="DuckDuckGo", variable=self.use_duckduckgo).grid(row=0, column=2, sticky="w", padx=20)
        ttk.Checkbutton(pref_frame, text="Wikipedia", variable=self.use_wikipedia).grid(row=0, column=3, sticky="w", padx=20)
        ttk.Checkbutton(pref_frame, text="Yahoo", variable=self.use_yahoo).grid(row=1, column=2, sticky="w", padx=20)
        
        self.file_frame = ttk.Frame(pref_frame)
        self.file_frame.grid(row=3, column=0, columnspan=4, sticky="w", pady=5)
        self.image_label = ttk.Label(self.file_frame)
        self.image_label.pack(side=tk.LEFT)
        self.remove_image_button = ttk.Button(self.file_frame, text="Remove File", command=self.remove_image)
        self.remove_image_button.pack(side=tk.LEFT)
        self.remove_image_button.pack_forget()
        
    def update_font_size(self):
        current_font = font.nametofont(self.chat_display.cget("font"))
        current_font.configure(size=self.font_size.get())
        self.chat_display.config(font=current_font)
        self.user_input.config(font=current_font)
        
    def send_message(self, event=None):
        user_message = self.user_input.get()
        self.user_input.delete(0, tk.END)
        self.update_chat_display(f"You: {user_message}" + (" [Image attached]" if self.current_image_base64 else ""))
        asyncio.create_task(self.process_message(user_message))
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            self.current_image_path = file_path
            image = Image.open(file_path)
            image.thumbnail((100, 100))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.remove_image_button.pack(side=tk.LEFT)
            self.update_chat_display(f"Image uploaded: {os.path.basename(file_path)}")
            
            with open(file_path, "rb") as image_file:
                self.current_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            self.image_counter += 1
            
    def remove_image(self):
        self.current_image_path = None
        self.current_image_base64 = None
        self.image_label.config(image='')
        self.remove_image_button.pack_forget()
        self.update_chat_display("Image removed")
        self.image_counter = 0
        
    def update_chat_display(self, message, highlight_header=False):
        self.chat_display.config(state=tk.NORMAL)
        if highlight_header:
            header, _, content = message.partition(': ')
            self.chat_display.insert(tk.END, header + ': ', 'orange')
            self.chat_display.insert(tk.END, content + '\n\n')
        else:
            self.chat_display.insert(tk.END, message + '\n\n')
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_history.append(message)
        
    async def process_message(self, prompt):
        start_time = asyncio.get_event_loop().time()
        
        chat_context = "\n".join(self.chat_history[-5:])
        
        try:
            search_results = await self.perform_searches(prompt)
            
            file_description = await self.analyze_image(prompt) if self.current_image_base64 else ""
            
            orchestrator = Orchestrator(self.models, self.user_preferences)
            final_output, explanation, layer_outputs = await orchestrator.process(prompt, chat_context, file_description, search_results)
            
            for layer, output in layer_outputs.items():
                self.update_chat_display(f"{layer.capitalize()} Output: {output}", highlight_header=True)
            
            self.update_chat_display(f"Final Output: {final_output}", highlight_header=True)
            self.update_chat_display(f"Explanation: {explanation}", highlight_header=True)
            
        except Exception as e:
            self.handle_error(e)
            
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        self.timer_label.config(text=f"Time: {processing_time:.1f}s")
        
    async def perform_searches(self, prompt):
        search_tasks = []
        if self.use_duckduckgo.get():
            search_tasks.append(search_duckduckgo_async(prompt))
        if self.use_wikipedia.get():
            search_tasks.append(search_wikipedia_async(prompt))
        if self.use_yahoo.get():
            search_tasks.append(search_yahoo_async(prompt))
            
        search_results = await asyncio.gather(*search_tasks)
        all_results = [result for sublist in search_results if sublist for result in sublist]
        return self.filter_relevant_results(all_results, prompt)
    
    def filter_relevant_results(self, results, query):
        if not results:
            return []
        query_words = set(query.lower().split())
        return [result for result in results if query_words & set(result['title'].lower().split())]
    
    async def analyze_image(self, prompt):
        if self.current_image_base64:
            llava_response = await self.generate_response("llava:13b", prompt, self.current_image_base64)
            self.update_chat_display(f"llava:13b: {llava_response}")
            return f"Image analysis: {llava_response}"
        return ""

    def handle_error(self, error):
        error_message = f"An error occurred: {str(error)}"
        self.update_chat_display(error_message)
        logging.error(f"Error in processing: {error}")
        self.update_chat_display("I'm having trouble processing your request. Please try again or rephrase your question.")
        
    async def generate_response(self, model, prompt, images=None):
        cache_key = f"{model}:{prompt}:{images}"
        cached_response = response_cache.get(cache_key)
        if cached_response:
            logging.info(f"Cache hit for {model}")
            return cached_response
        
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "max_tokens": 2048
            }
            if images and model == "llava:13b":
                payload["images"] = [images]
            headers = {'Content-Type': 'application/json'}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        response_text = response_json.get('response', '')
                    else:
                        error_text = await response.text()
                        logging.error(f"Error: {response.status} - {error_text}")
                        return None
            response_cache.set(cache_key, response_text)
            return response_text
        except Exception as e:
            logging.error(f"Error in API call for model {model}: {e}")
            return None

    def reset_chat(self):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_history.clear()
        self.remove_image()
        self.update_chat_display("Chat has been reset.")
        
class Orchestrator:
    def __init__(self, models, user_preferences):
        self.models = models
        self.user_preferences = user_preferences
        self.api_base_url = "http://localhost:11434/api"  # Adjust this to your actual API endpoint
        
    async def process(self, prompt, chat_context, file_description, search_results):
        layer_outputs = {}
        for layer in ["layer1", "layer2"]:
            layer_output = await self.process_layer(layer, prompt, chat_context, file_description, search_results, layer_outputs)
            if layer_output:
                layer_outputs[layer] = layer_output
            else:
                logging.warning(f"No output generated for {layer}")
                
        if not layer_outputs:
            return "I'm sorry, but I couldn't generate a response at this time.", "All layers failed to produce output.", {}
        
        final_output, explanation = await self.generate_final_response(prompt, layer_outputs)
        return final_output, explanation, layer_outputs
    
    async def process_layer(self, layer, prompt, chat_context, file_description, search_results, previous_outputs):
        models = self.dynamic_model_selection(layer, prompt)
        combined_prompt = self.create_layer_prompt(layer, prompt, chat_context, file_description, search_results, previous_outputs)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            layer_output = await self.run_models(models, combined_prompt)
            if layer_output:
                quality_score = await self.evaluate_output_quality(layer_output, prompt)
                
                if quality_score >= 0.3:
                    logging.info(f"{layer} output quality score: {quality_score:.2f}")
                    return layer_output
                elif attempt < max_attempts - 1:
                    logging.info(f"Low quality output from {layer}. Score: {quality_score:.2f}. Retrying... (Attempt {attempt + 1})")
                    combined_prompt = self.refine_prompt(combined_prompt, quality_score)
                    models = self.dynamic_model_selection(layer, prompt, exclude=models)
            else:
                logging.warning(f"No output from {layer} on attempt {attempt + 1}")
                
        logging.warning(f"Suboptimal output from {layer} after {max_attempts} attempts.")
        return layer_output or f"No satisfactory response could be generated for {layer}."
    
    def dynamic_model_selection(self, layer, prompt, exclude=None):
        available_models = [m for m in self.models[layer] if m not in (exclude or [])]
        
        if "question" in prompt.lower():
            return [m for m in available_models if "mistral" in m or "llama" in m] or available_models
        elif "image" in prompt.lower():
            return [m for m in available_models if "llava" in m] or available_models
        else:
            return random.sample(available_models, min(2, len(available_models)))
        
    def create_layer_prompt(self, layer, prompt, chat_context, file_description, search_results, previous_outputs):
        style = self.user_preferences["style"].get()
        detail_level = self.user_preferences["detail_level"].get()
        
        return f"""
        Style: {style}
        Detail Level: {detail_level}
        
        Chat history:
        {chat_context}
        
        User's question: {prompt}
        
        {file_description}
        
        Search results:
        {self.format_search_results(search_results)}
        
        Previous layer outputs:
        {' '.join(previous_outputs.values())}
        
        Please provide a {style} response with {detail_level} detail level.
        """
    
    def format_search_results(self, results):
        return "\n".join([f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}" for r in results])
    
    async def evaluate_output_quality(self, output, prompt):
        if not output.strip():
            return 0.0
        
        relevance_score = self.calculate_relevance(output, prompt)
        coherence_score = self.calculate_coherence(output)
        length_score = min(1.0, len(output.split()) / 100)  # Favor responses with at least 100 words
        
        total_score = (relevance_score + coherence_score + length_score) / 3
        return total_score
    
    def calculate_relevance(self, output, prompt):
        prompt_words = set(prompt.lower().split())
        output_words = set(output.lower().split())
        overlap = len(prompt_words.intersection(output_words))
        return min(1.0, overlap / len(prompt_words))
    
    def calculate_coherence(self, output):
        words = output.split()
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def refine_prompt(self, prompt, quality_score):
        return prompt + f"\nThe previous response did not meet our quality standards (score: {quality_score:.2f}). Please provide a more relevant, coherent, and detailed response."
    
    async def generate_final_response(self, original_prompt, layer_outputs):
        final_prompt = f"""
        Based on the user's original question: "{original_prompt}" and the following layer outputs: 
        {' '.join(layer_outputs.values())}
        
        Please provide:
        1. A concise, clear, and user-friendly response.
        2. An explanation of how you arrived at this response.
        
        Focus on directly answering the user's question while considering the chat history and user preferences.
        """
        
        try:
            response = await self.generate_response(self.models["final"], final_prompt)
            parts = response.split("Explanation:", 1)
            answer = parts[0].strip()
            explanation = parts[1].strip() if len(parts) > 1 else "No explanation provided."
            return answer, explanation
        except Exception as e:
            logging.error(f"Error in final response generation: {e}")
            return "I encountered an error while processing your request.", "Error occurred during processing."
        
    async def generate_response(self, model, prompt, images=None):
        try:
            url = f"{self.api_base_url}/generate"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "max_tokens": 2048
            }
            if images and model == "llava:13b":
                payload["images"] = images
                
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        error_text = await response.text()
                        logging.error(f"API error for model {model}: Status {response.status}, Response: {error_text}")
                        return None
        except Exception as e:
            logging.error(f"Error generating response from {model}: {e}")
            return None
        
    async def run_models(self, models, prompt, images=None):
        max_parallel = 3
        tasks = [self.generate_response(model, prompt, images if images and model == "llava:13b" else None) for model in models[:max_parallel]]
        responses = await asyncio.gather(*tasks)
        valid_responses = [r for r in responses if r]
        if valid_responses:
            return " ".join(valid_responses)
        else:
            logging.warning(f"No valid responses from any model for prompt: {prompt[:50]}...")
            return None
        
async def main():
    root = tk.Tk()
    app = AdvancedChatUI(root)
    
    while True:
        root.update()
        await asyncio.sleep(0.01)  # Reduced sleep time for better responsiveness
        
if __name__ == "__main__":
    asyncio.run(main())