from src.chatbots_with_transformers.facebook_blender import chat_with_facebook_blenderbot
from src.chatbots_with_transformers.google_flan_t5 import chat_with_google_flanT5bot

def main():
    print("Choose a chatbot:")
    print("1. Facebook BlenderBot")
    print("2. Google FLAN-T5")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        print("\nStarting Facebook BlenderBot...\n")
        chat_with_facebook_blenderbot()
    elif choice == "2":
        print("\nStarting Google FLAN-T5 Bot...\n")
        chat_with_google_flanT5bot()
    else:
        print("Invalid choice. Please run the program again and select 1 or 2.")

if __name__ == "__main__":
    main()