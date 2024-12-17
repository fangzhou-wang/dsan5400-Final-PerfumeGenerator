import pandas as pd
import re
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from itertools import chain
from nltk.stem import WordNetLemmatizer

class PerfumeRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, encoding='latin1', delimiter=';')
        self.main_accord_trees = {}
        self.gender_tree = None
        self.detailed_tree = None
        self.gender_le = LabelEncoder()
        self.mlb = None
        self.X_gender = None
        self.X_detailed = None
        self.columns_to_clean = ['Perfume', 'Brand', 'Country', 'Gender', 'Top', 'Middle', 'Base', \
                                 'mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5']

    def clean_text(self, text):
        lemmatizer = WordNetLemmatizer()
        if isinstance(text, float):
            text = ""
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
        return text

    def preprocess(self):
        for col in self.columns_to_clean:
            self.df[col] = self.df[col].apply(self.clean_text)

    def build_main_accord_trees(self):
        accord_columns = ['mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'mainaccord5']
        for i in range(len(accord_columns) - 1):
            input_col = accord_columns[:i+1]
            target_col = accord_columns[i+1]

            df_subset = self.df.dropna(subset=input_col + [target_col])
            X = pd.get_dummies(df_subset[input_col])
            y = df_subset[target_col]

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            tree = DecisionTreeClassifier(random_state=42)
            tree.fit(X_train, y_train)

            print(f"Decision Tree for {target_col} based on {', '.join(input_col)}")
            print(export_text(tree, feature_names=X.columns.tolist()))

            self.main_accord_trees[target_col] = (tree, le)

    def build_detailed_tree(self):
        def clean_and_split_accord(text):
            """
            Cleans and splits accord strings into individual phrases or words.
            Retains valid phrases like 'fruity notes' and individual words.
            """
            if not isinstance(text, str):
                return []
            # Split by commas, clean non-alphabetic characters, and keep valid entries
            words = [re.sub(r'[^a-zA-Z-\s]', '', word.strip()) for word in text.split(',')]
            return [word for word in words if len(word) > 1]  # Exclude single letters or empty strings

        # Extract individual phrases/words from Top, Middle, and Base columns
        self.df['Top_cleaned'] = self.df['Top'].dropna().apply(clean_and_split_accord)
        self.df['Middle_cleaned'] = self.df['Middle'].dropna().apply(clean_and_split_accord)
        self.df['Base_cleaned'] = self.df['Base'].dropna().apply(clean_and_split_accord)

        # Combine all unique detailed accords
        detailed_accords = list(set(chain.from_iterable(
            self.df['Top_cleaned'].dropna() + self.df['Middle_cleaned'].dropna() + self.df['Base_cleaned'].dropna()
        )))

        # Process each row into a list of individual words/phrases for detailed accords
        self.mlb = MultiLabelBinarizer(classes=detailed_accords)
        self.df['detailed_accord'] = self.df[['Top_cleaned', 'Middle_cleaned', 'Base_cleaned']].apply(
            lambda row: list(set(chain.from_iterable(row.dropna()))), axis=1
        )

        # Binarize the detailed accord column for model training
        y_detailed = self.mlb.fit_transform(self.df['detailed_accord'])

        # Use mainaccord1 as the feature for predicting detailed accords
        self.X_detailed = pd.get_dummies(self.df['mainaccord1'])

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(self.X_detailed, y_detailed, test_size=0.2, random_state=42)

        # Train the decision tree
        self.detailed_tree = DecisionTreeClassifier(random_state=42)
        self.detailed_tree.fit(X_train, y_train)

        print("Decision Tree for Detailed Accords based on Main Accord 1:")
        print(export_text(self.detailed_tree, feature_names=self.X_detailed.columns.tolist()))

    def build_gender_tree(self):
        y_gender = self.gender_le.fit_transform(self.df['Gender'])
        self.X_gender = pd.get_dummies(self.df[['mainaccord1', 'mainaccord2']])

        X_train, X_test, y_train, y_test = train_test_split(self.X_gender, y_gender, test_size=0.2, random_state=42)

        self.gender_tree = DecisionTreeClassifier(random_state=42)
        self.gender_tree.fit(X_train, y_train)

        print("Decision Tree for Gender based on Main Accord 1 and 2:")
        print(export_text(self.gender_tree, feature_names=self.X_gender.columns.tolist()))

    def recommend_main_accord(self, selected_main_accords):
        """
        Recommends the next main accord options based on selected main accords.
        """
        if not selected_main_accords:  # Initial case: return all unique mainaccord1 values
            return self.df['mainaccord1'].dropna().unique()

        current_level = len(selected_main_accords)
        if current_level >= 5:  # No further levels beyond mainaccord5
            return "No further main accords to recommend."

        # Get the column for the next main accord
        next_main_accord_col = f'mainaccord{current_level + 1}'

        # Filter the dataset based on selected main accords
        filter_conditions = True
        for level, accord in selected_main_accords.items():
            filter_conditions &= self.df[level] == accord

        # Get unique values for the next main accord
        next_options = self.df.loc[filter_conditions, next_main_accord_col].dropna().unique()
        if len(next_options) == 0:
            return "No further main accords to recommend."

        return next_options

    def recommend_detailed_accord(self, selected_main_accords, selected_detailed_accord=None):
        """
        Recommends detailed accords based on the selected main accord combination.
        Only returns detailed accords that appear in rows with the selected main accord combination.
        """
        # Step 1: Filter rows based on selected main accords
        filter_conditions = True
        for level, accord in selected_main_accords.items():
            filter_conditions &= (self.df[level] == accord)

        filtered_rows = self.df[filter_conditions]

        # Debugging: Print filtered rows for main accords
        print("Filtered Rows for Main Accords:")
        print(filtered_rows[['mainaccord1', 'mainaccord2', 'mainaccord3', 'detailed_accord']].head())

        # Step 2: Extract unique detailed accords from filtered rows
        detailed_matches = list(set(chain.from_iterable(filtered_rows['detailed_accord'].dropna())))

        # Step 3: If a detailed accord is selected, recommend co-occurring accords
        if selected_detailed_accord:
            filtered_rows = filtered_rows[
                filtered_rows['detailed_accord'].apply(lambda accords: selected_detailed_accord in accords if accords else False)
            ]

            # Debugging: Print rows containing the selected detailed accord
            print("Filtered Rows for Detailed Accord:")
            print(filtered_rows[['mainaccord1', 'mainaccord2', 'mainaccord3', 'detailed_accord']].head())

            co_occurring_detailed_matches = list(set(chain.from_iterable(filtered_rows['detailed_accord'].dropna())))
            co_occurring_detailed_matches.remove(selected_detailed_accord)
            return sorted(co_occurring_detailed_matches)

        return sorted(detailed_matches)  # Return unique, sorted accords

    def recommend_perfumes(self, selected_main_accords, detailed_accord=None, gender=None):
        """
        Recommend perfumes based on selected main accords, detailed accords, and gender.
        Returns only the perfume name and brand.
        """
        # Step 1: Filter rows based on main accords
        filter_conditions = True
        for level, accord in selected_main_accords.items():
            filter_conditions &= (self.df[level] == accord)

        filtered_rows = self.df[filter_conditions]

        # Debugging: Print rows after filtering by main accords
        print("Filtered Rows for Main Accords:")
        print(filtered_rows[['Perfume', 'Brand', 'mainaccord1', 'mainaccord2', 'mainaccord3', 'detailed_accord']].head())

        # Step 2: Filter rows by detailed accord if provided
        if detailed_accord:
            filtered_rows = filtered_rows[
                filtered_rows['detailed_accord'].apply(lambda accords: detailed_accord in accords if accords else False)
            ]

            # Debugging: Print rows after filtering by detailed accord
            print("Filtered Rows for Detailed Accord:")
            print(filtered_rows[['Perfume', 'Brand', 'mainaccord1', 'mainaccord2', 'mainaccord3', 'detailed_accord']].head())

        # Step 3: Filter rows by gender if provided
        if gender:
            filtered_rows = filtered_rows[filtered_rows['Gender'] == gender]

        # Step 4: Return perfume recommendations
        if filtered_rows.empty:
            return "No perfumes found matching the specified criteria."

        # Return only the perfume name and brand
        return filtered_rows[['Perfume', 'Brand']]


# Example User Interaction Script for .py File Execution
if __name__ == "__main__":
    recommender = PerfumeRecommender('./fragrantica-com-fragrance-dataset/fra_cleaned.csv')
    recommender.preprocess()
    recommender.build_main_accord_trees()
    recommender.build_detailed_tree()
    recommender.build_gender_tree()

    # Step 1: User selects main accords
    selected_accords = {}
    while True:
        next_main_accords = recommender.recommend_main_accord(selected_accords)
        if isinstance(next_main_accords, str):  # Check if no further recommendations
            print(next_main_accords)
            break
        print("\nAvailable Main Accord Options:", next_main_accords)
        user_choice = input("Choose a main accord or type 'complete' to finish: ").strip()
        if user_choice.lower() == 'complete':
            if not selected_accords:  # Prevent the user from completing without selecting a main accord
                print("You must select at least one main accord to proceed.")
                continue
            break
        elif user_choice in next_main_accords:
            selected_accords[f'mainaccord{len(selected_accords) + 1}'] = user_choice
        else:
            print("Invalid choice. Please choose from the available options.")

    # Step 2: User selects detailed accords
    # Logic to generate detailed accords after the user completes main accord selection
    if selected_accords:
        print("\nSelected Main Accords:", selected_accords)
        detailed_options = recommender.recommend_detailed_accord(selected_accords)
        if detailed_options:  # If detailed accords are available
            print("\nAvailable Detailed Accord Options:", detailed_options)
            detailed_choice = input("Choose a detailed accord or type 'complete' to skip: ").strip()
            if detailed_choice.lower() == 'complete':
                detailed_choice = None
            elif detailed_choice not in detailed_options:
                print("Invalid choice. No detailed accord will be selected.")
                detailed_choice = None
        else:
            print("No detailed accords available for the selected main accords.")
            detailed_choice = None
    else:
        detailed_choice = None

    # Step 3: User selects gender
    gender_choice = input("\nDo you want to filter by gender? Type 'men', 'women', 'unisex', or 'none': ").strip()
    if gender_choice.lower() == 'none':
        gender_choice = None

    # Step 4: Get Recommendations
    recommendations = recommender.recommend_perfumes(
        selected_main_accords=selected_accords,
        detailed_accord=detailed_choice,
        gender=gender_choice
    )
    print("\nRecommended Perfumes:")
    print(recommendations)






