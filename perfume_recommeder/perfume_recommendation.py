import pandas as pd
import re
import logging
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from itertools import chain
from nltk.stem import WordNetLemmatizer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerfumeRecommender:
    def __init__(self, perfume_data):
        self.df = perfume_data
        self.main_accord_trees = {}
        self.gender_tree = None
        self.detailed_tree = None
        self.gender_le = LabelEncoder()
        self.mlb = None

    @staticmethod
    def clean_text(text):
        lemmatizer = WordNetLemmatizer()
        if isinstance(text, float):
            text = ""
        text = re.sub(r"\\W", " ", text)
        text = re.sub(r"\\d+", "", text)
        text = text.lower()
        return " ".join(lemmatizer.lemmatize(word) for word in text.split())

    def build_main_accord_trees(self):
        logger.info("Building main accord decision trees...")
        accord_columns = ["mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"]
        for i in range(len(accord_columns) - 1):
            input_col = accord_columns[: i + 1]
            target_col = accord_columns[i + 1]

            try:
                df_subset = self.df.dropna(subset=input_col + [target_col])
                X = pd.get_dummies(df_subset[input_col])
                y = df_subset[target_col]

                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
                tree = DecisionTreeClassifier(random_state=42)
                tree.fit(X_train, y_train)
                self.main_accord_trees[target_col] = (tree, le)
                logger.info(f"Successfully built tree for predicting {target_col}.")
            except Exception as e:
                logger.error(f"Error building main accord tree for {target_col}: {e}")

    def build_detailed_tree(self):
        logger.info("Building detailed accord decision tree...")
        def clean_and_split_accord(text):
            if not isinstance(text, str):
                return []
            words = [re.sub(r"[^a-zA-Z-\\s]", "", word.strip()) for word in text.split(",")]
            return [word for word in words if len(word) > 1]

        try:
            self.df["detailed_accord"] = self.df[["Top", "Middle", "Base"]].apply(
                lambda row: list(set(chain.from_iterable([clean_and_split_accord(x) for x in row.dropna()]))), axis=1
            )

            self.mlb = MultiLabelBinarizer()
            y_detailed = self.mlb.fit_transform(self.df["detailed_accord"])
            X = pd.get_dummies(self.df["mainaccord1"])

            X_train, _, y_train, _ = train_test_split(X, y_detailed, test_size=0.2, random_state=42)
            self.detailed_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
            logger.info("Successfully built detailed accord decision tree.")
        except Exception as e:
            logger.error(f"Error building detailed accord tree: {e}")

    def build_gender_tree(self):
        logger.info("Building gender decision tree...")
        try:
            y_gender = self.gender_le.fit_transform(self.df["Gender"].fillna("Unknown"))
            X_gender = pd.get_dummies(self.df[["mainaccord1", "mainaccord2"]])

            X_train, _, y_train, _ = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42)
            self.gender_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
            logger.info("Successfully built gender decision tree.")
        except Exception as e:
            logger.error(f"Error building gender tree: {e}")

    def recommend_main_accord(self, selected_main_accords):
        logger.info("Recommending next main accord...")
        current_level = len(selected_main_accords)
        if current_level >= 5:
            logger.info("All main accord levels have been selected.")
            return "No further main accords to recommend."

        next_main_accord_col = f"mainaccord{current_level + 1}"
        filter_conditions = True
        for level, accord in selected_main_accords.items():
            filter_conditions &= self.df[level] == accord

        next_options = self.df.loc[filter_conditions, next_main_accord_col].dropna().unique()
        if next_options.size > 0:
            logger.info(f"Recommended next main accord: {next_options}")
            return sorted(next_options)
        else:
            logger.warning("No further main accords to recommend.")
            return "No further main accords to recommend."

    def recommend_detailed_accord(self, selected_main_accords):
        logger.info("Recommending detailed accords...")
        filter_conditions = True
        for level, accord in selected_main_accords.items():
            filter_conditions &= self.df[level] == accord

        filtered_rows = self.df[filter_conditions]
        detailed_options = list(set(chain.from_iterable(filtered_rows["detailed_accord"].dropna())))

        if detailed_options:
            logger.info(f"Recommended detailed accords: {detailed_options}")
            return sorted(detailed_options)
        else:
            logger.warning("No detailed accords available.")
            return "No detailed accords available."

    def recommend_perfumes(self, selected_main_accords, selected_detailed_accord=None, gender=None):
        logger.info("Recommending perfumes...")
        try:
            filter_conditions = True
            for level, accord in selected_main_accords.items():
                filter_conditions &= self.df[level] == accord

            if selected_detailed_accord:
                filter_conditions &= self.df["detailed_accord"].apply(
                    lambda accords: selected_detailed_accord in accords if accords else False
                )

            if gender:
                filter_conditions &= self.df["Gender"] == gender

            recommendations = self.df[filter_conditions]
            if not recommendations.empty:
                logger.info(f"Found {len(recommendations)} matching perfumes.")
                return recommendations[["Perfume", "Brand"]]
            else:
                logger.warning("No perfumes found matching the criteria.")
                return "No perfumes found."
        except Exception as e:
            logger.error(f"Error during perfume recommendation: {e}")
            return "Error occurred while recommending perfumes."