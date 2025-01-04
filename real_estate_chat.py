from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.vector_stores import MongoDBAtlasVectorSearch
from llama_index.embeddings import OpenAIEmbedding
from pymongo import MongoClient
from typing import List, Dict, Any, Optional
import datetime
import os
import json
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class RealEstateQuerySystem:
    def __init__(self):
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(os.getenv('MONGODB_URI'))
        self.db = self.mongo_client['real_estate']
        self.properties_collection = self.db['properties']
        self.query_history = self.db['query_history']
        self.vector_store = self.db['vector_store']
        
        # Initialize OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding()
        
        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            mongodb_client=self.mongo_client,
            db_name="real_estate",
            collection_name="vector_store",
            index_name="vector_index", 
        )
        
        # Initialize vector index
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
    def init_database(self):
        """Initialize database with required indexes"""
        # Create text indexes for searching
        self.properties_collection.create_index([
            ("location.city", "text"),
            ("location.area", "text"),
            ("location.locality", "text"),
            ("description", "text")
        ])
        
        # Create indexes for query history
        self.query_history.create_index([("timestamp", -1)])
        self.query_history.create_index([("context.location", 1)])

    def add_property(self, property_data: Dict[str, Any]):
        """Add a new property to the database"""
        property_doc = {
            "property_id": str(property_data.get("property_id")),
            "type": property_data.get("type"),
            "location": {
                "city": property_data.get("city"),
                "area": property_data.get("area"),
                "locality": property_data.get("locality"),
                "coordinates": property_data.get("coordinates")
            },
            "specifications": {
                "bhk": property_data.get("bhk"),
                "size": property_data.get("size"),
                "price": property_data.get("price"),
                "amenities": property_data.get("amenities", [])
            },
            "description": property_data.get("description"),
            "created_at": datetime.datetime.utcnow()
        }
        
        # Insert into MongoDB
        self.properties_collection.insert_one(property_doc)
        
        # Create document for vector store
        doc_text = f"""
        Property Type: {property_doc['type']}
        Location: {property_doc['location']['city']}, {property_doc['location']['area']}, {property_doc['location']['locality']}
        Specifications: {property_doc['bhk']} BHK, {property_doc['specifications']['size']} sq ft
        Price: {property_doc['specifications']['price']}
        Description: {property_doc['description']}
        """
        
        doc = Document(text=doc_text, id_=property_doc['property_id'])
        self.index.insert(doc)

    def extract_context(self, query: str) -> Dict[str, Any]:
        """Extract context parameters from the natural language query"""
        # Use OpenAI to extract entities and context
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Extract real estate search parameters from the query. Include location, property type, and specifications."
            }, {
                "role": "user",
                "content": query
            }]
        )
        
        # Parse the response and return structured context
        try:
            context = json.loads(response.choices[0].message.content)
        except:
            context = {
                "location": [],
                "property_type": None,
                "specifications": {}
            }
        
        return context

    def generate_mongodb_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate MongoDB query from context"""
        query = {}
        
        if context.get("location"):
            location_query = []
            for loc in context["location"]:
                location_query.extend([
                    {"location.city": {"$regex": loc, "$options": "i"}},
                    {"location.area": {"$regex": loc, "$options": "i"}},
                    {"location.locality": {"$regex": loc, "$options": "i"}}
                ])
            query["$or"] = location_query
            
        if context.get("property_type"):
            query["type"] = {"$regex": context["property_type"], "$options": "i"}
            
        if context.get("specifications"):
            for key, value in context["specifications"].items():
                if key == "bhk":
                    query["specifications.bhk"] = value
                elif key == "price":
                    query["specifications.price"] = {"$lte": value}
                elif key == "size":
                    query["specifications.size"] = {"$gte": value}
                    
        return query

    def modify_query(self, previous_query: Dict[str, Any], new_context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify previous query based on new context"""
        if not previous_query:
            return self.generate_mongodb_query(new_context)
            
        modified_query = previous_query.copy()
        
        # Update location constraints
        if new_context.get("location"):
            if "$or" in modified_query:
                existing_locations = set()
                for loc_query in modified_query["$or"]:
                    for value in loc_query.values():
                        if isinstance(value, dict) and "$regex" in value:
                            existing_locations.add(value["$regex"])
                            
                new_locations = set(new_context["location"]) - existing_locations
                for loc in new_locations:
                    modified_query["$or"].extend([
                        {"location.city": {"$regex": loc, "$options": "i"}},
                        {"location.area": {"$regex": loc, "$options": "i"}},
                        {"location.locality": {"$regex": loc, "$options": "i"}}
                    ])
            else:
                modified_query.update(self.generate_mongodb_query({"location": new_context["location"]}))
                
        # Update other constraints
        for key, value in new_context.items():
            if key != "location" and value:
                modified_query.update(self.generate_mongodb_query({key: value}))
                
        return modified_query

    def execute_hybrid_search(self, query: Dict[str, Any], natural_query: str) -> Dict[str, Any]:
        """Execute hybrid search using both MongoDB and RAG"""
        # First, try RAG-based search
        query_engine = self.index.as_query_engine()
        rag_response = query_engine.query(natural_query)
        
        # Get MongoDB results
        mongo_results = list(self.properties_collection.find(query))
        
        # Combine results
        combined_response = {
            "rag_response": str(rag_response),
            "mongodb_results": mongo_results,
            "query_type": "hybrid"
        }
        
        return combined_response

    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """Main function to process natural language queries"""
        # Extract context from query
        context = self.extract_context(natural_language_query)
        
        # Get most recent relevant query
        previous_query = self.query_history.find_one(
            {'context.location': {'$in': context.get('location', [])}},
            sort=[('timestamp', -1)]
        )
        
        # Generate or modify query
        if previous_query:
            mongodb_query = self.modify_query(previous_query['generated_query'], context)
        else:
            mongodb_query = self.generate_mongodb_query(context)
        
        # Store query in history
        self.query_history.insert_one({
            'natural_language_query': natural_language_query,
            'generated_query': mongodb_query,
            'context': context,
            'timestamp': datetime.datetime.utcnow()
        })
        
        # Execute hybrid search
        results = self.execute_hybrid_search(mongodb_query, natural_language_query)
        
        return results

# Example usage
def main():
    # Initialize the system
    query_system = RealEstateQuerySystem()
    query_system.init_database()
    
    # Add sample property
    sample_property = {
        "property_id": "123",
        "type": "Apartment",
        "city": "Pune",
        "area": "Wakad",
        "locality": "Blue Ridge",
        "coordinates": {"lat": 18.5793, "lng": 73.7765},
        "bhk": 2,
        "size": 1200,
        "price": 8500000,
        "amenities": ["Swimming Pool", "Gym", "Park"],
        "description": "Beautiful 2 BHK apartment in Blue Ridge, Wakad with modern amenities"
    }
    query_system.add_property(sample_property)
    
    # Test queries
    queries = [
        "Show me properties in Pune",
        "Show me properties in Wakad",
        "Show me 2BHK properties in Wakad under 1 crore"
    ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        results = query_system.process_query(query)
        print(f"Results: {json.dumps(results, indent=2, default=str)}")

if __name__ == "__main__":
    main()