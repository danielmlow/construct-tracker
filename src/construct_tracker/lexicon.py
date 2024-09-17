"""Create for given constructs using Generative AI and count lexicons in documents.
Author: Daniel M. Low
License: Apache 2.0.
"""

import datetime
import inspect
import json
import os
import pickle
import random
import re
import string
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import dill
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from tqdm import tqdm

# Local
from .genai import api_request  # local
from .utils import lemmatizer  # local / construct-tracker package
from .utils.logger_config import setup_logger
from .utils.word_count import word_count  # local

# Set up the logger
logger = setup_logger()


def generate_variable_name(variable_str: str) -> str:
	"""Replace spaces with underscores, lowercase the string, and remove certain punctuation.

	Args:
					variable_str: The string to convert.

	Returns:
					The converted string.
	"""
	variable_name = (
		variable_str.replace(",", "")
		.replace(" & ", "_")
		.replace(" and ", "_")
		.replace(" ", "_")
		.replace("-", "_")
		.lower()
	)
	return variable_name


def generate_prompt(
	construct: str,
	prompt_name: Optional[str] = None,
	prompt: str = "default",
	domain: Optional[str] = None,
	definition: Optional[str] = None,
	examples: Optional[Union[List[str], str]] = None,
	output_format: str = "default",
	remove_parentheses_definition: bool = True,
) -> str:
	"""Generate a prompt based on provided parameters.

	Args:
					construct: The construct for which the prompt is generated.
					prompt_name: Optional; the name of the prompt.
					prompt: Optional; the prompt template.
					domain: Optional; the domain of the construct.
					definition: Optional; the definition of the construct.
					examples: Optional; examples related to the construct.
					output_format: Optional; the format of the output.
					remove_parentheses_definition: Optional; whether to remove parentheses from the definition.

	Returns:
					The generated prompt.
	"""
	if output_format == "default":
		output_format = (
			"Each token should be separated by a semicolon. Do not return duplicate tokens. Do not provide any"
			" explanation or additional text beyond the tokens."
		)
	elif output_format == "json":
		output_format = (
			"Provide tokens in JSON output. Do not return duplicate tokens. Do not provide any explanation or"
			" additional text beyond the tokens."
		)

	if not isinstance(prompt_name, str):
		prompt_name = construct.replace("_", " ").lower()

	if prompt == "default":
		prompt = (
			"Provide many single words and some short phrases (all in lowercase unless the word is generally in"
			" uppercase) related to"
		)
		if domain:
			domain = f"(in the {domain} domain). "
			prompt = f"""{prompt} {prompt_name} {domain}{output_format}"""
		else:
			prompt = f"""{prompt} {prompt_name}. {output_format}"""
		if definition:
			if remove_parentheses_definition:
				definition = re.sub(r"\(.*?\)", "", definition.lower().strip())
			prompt += f"\nHere is a definition of {prompt_name}: {definition}"

		if isinstance(examples, list):
			examples = "; ".join(examples)
		if isinstance(examples, str):
			prompt += f"\nHere are some examples (include these in the list): {examples}"

	return prompt


def find_partial_matching_strings(list_a: List[str], list_b: List[str]) -> Tuple[List[str], Dict[str, str]]:
	"""Find strings in list_a that contain any of the strings in list_b, excluding exact matches.

	Args:
					list_a: List of strings to search within.
					list_b: List of substrings to search for.

	Returns:
					A tuple of lists containing partial matches and matched substrings.
	"""
	partial_matching_strings = []
	matched_substrings = {}
	for string_a in list_a:
		for string_b in list_b:
			if string_b in string_a and string_a != string_b:
				partial_matching_strings.append(string_a)
				matched_substrings[string_a] = string_b
	return partial_matching_strings, matched_substrings


def remove_substrings(s: str, substrings: List[str]) -> str:
	"""Remove specified substrings from a string.

	Args:
					s: The string from which substrings are removed.
					substrings: List of substrings to remove.

	Returns:
					The modified string.
	"""
	for substring in substrings:
		s = s.replace(substring, "")
	return s


class Lexicon:
	"""A class to manage and manipulate lexicons."""

	def __init__(
		self,
		name: Optional[str] = None,
		description: Optional[str] = None,
		version: str = "1.0",
		creator: Optional[str] = None,
	) -> None:
		"""Initializes the class with optional name, description, and other attributes."""
		self.name = name
		self.description = description
		self.version = version
		self.creator = creator
		self.constructs = {}
		self.construct_names = []
		if isinstance(name, str):
			self.constructs = load_lexicon(name)
		self.exact_match_n = 4
		self.exact_match_tokens = []
		self.remove_from_all_constructs = []
		self.attributes = {}

	def set_attribute(self, key: str, value: Any) -> None:
		"""Set an attribute for the lexicon."""
		self.attributes[key] = value

	def get_attribute(self, key: str, default: Optional[Any] = None) -> Any:
		"""Get an attribute from the lexicon."""
		return self.attributes.get(key, default)

	def generate_definition(
		self,
		construct: str,
		model: str = "command-nightly",
		domain: Optional[str] = None,
		timeout: int = 45,
		num_retries: int = 2,
	) -> Tuple[str, str]:
		"""Generate a brief definition of a construct using a specified model."""
		if domain:
			prompt = f"""Please provide a brief definition of {construct} (in the {domain} domain). Provide reference in APA format where you got it from. Return result in the following string format:'''{{"{construct}": "the_definition", "reference":"the_reference"}}'''"""
			# TODO: the formatting request might not work well for models worse than GPT-4
		else:
			prompt = f"Provide a brief definition of {construct}. Provide reference in APA format where you got it from. Return result in the following format: {{'{construct}': 'the_definition', 'reference':'the_reference'}}"
		definition = api_request(prompt, model=model, timeout=timeout, num_retries=num_retries)
		# Convert the string to a dictionary
		try:
			definition = definition = eval(eval(definition))
			reference = f"{model} which references: {definition['reference']}"
			definition = definition[construct]
		except ValueError as e:
			print(f"Error parsing string: {e}. Will try by splitting")
			reference = f"{model} which references: {definition.split('reference')[1]}"
			definition = definition.split('"reference"')[0]
			print(f"New definition: {definition}")
			print(f"Reference: {reference}")
			# definition will remain a string

		return definition, reference

	def clean_response(self, response: str, response_type: str = "tokens") -> Union[List[str], str]:
		"""Clean the response generated by the AI model.

		Args:
						response: The raw response string.
						response_type: Optional; the type of response, e.g., 'tokens' or 'definition'.

		Returns:
						The cleaned response.
		"""
		# response = "Here is a list of tokens related to sexual abuse and harassment, separated by semicolons: violated; abused; assaulted; raped; molested; harasses; nonconsensual; harassment; victimized; stalking; groping; coerced; profaned; derided; violated my boundaries. "
		# response = 'Acceptance and Commitment Therapy (ACT)'
		if response_type == "tokens":
			# response = lexicon.constructs['sexual_abuse_harassment']['tokens']
			# response = "suicide; kill; death; destroy; suffer; distress; desperate; ceased; expire; hurt; isolate; lonely; agonize; pain; anguish; grief; slit; bleed; hang; overdose; freeze to death; jump; fall; throttle; immolate; lie in traffic; hang oneself; jump off a bridge; jump in front of a train; drug overdose; hanging; drowning; suffocation; bullet wound; glass throat; carbon monoxide; running in front of a train; jumping from a building; steering into oncoming traffic; laying in the middle of the railroad tracks; tying a rope; waking up with a knife next to one's throat; planning; preparing; time; place; method; final; irreversible."
			response_list = response.split(";")

			tokens = []
			try:
				for t in response_list:
					if t == "":
						continue
					elif isinstance(t, str) and len(t) > 0:
						if "\n" in t:
							t = t.split("\n")
							# print(t)

							t = [n[0].lower() + n[1:] if (not n.isupper()) else n for n in t]
							tokens.extend(t)
						else:
							if t.isupper() or t[:2] == "I " or t[:2] == "I'":
								tokens.append(t)
							else:
								tokens.append(t[0].lower() + t[1:])
					else:
						logger.warning(f"token is either not str or some other issue: '{t}'. Not adding to lexicon.")
				tokens = [n.strip() for n in tokens]
				tokens_new = []
				for token in tokens:
					if "(" in t and ")" in t:
						parentheses_content = re.findall(r"\((.*?)\)", t)
						tokens_new.extend(parentheses_content)  # extend because it returns a list
						without_parentheses_content = re.sub(r"\(.*?\)", "", t)
						tokens_new.append(without_parentheses_content)  # remove .lower() here
					else:
						tokens_new.append(token)

				tokens = tokens_new.copy()
				remove_if_contains_substrings = ["single words", "here is a list"]
				tokens = [item for item in tokens if all(sub not in item for sub in remove_if_contains_substrings)]

				tokens = [n.strip().strip(".:;,!?") for n in tokens]  # TODO fix
				tokens = [n for n in tokens if n != ""]
				tokens = [
					n.replace("-", " ") if n.count("-") > 1 else n for n in tokens
				]  # for tokens like i-want-to-exit-life
				tokens = list(np.unique(tokens))  # Remove duplicates
				return tokens
			except (AttributeError, ValueError, TypeError) as e:
				logger.warning(
					f"{e} error. Gen AI response could not be parsed. Continuing with the raw response as a single token. This is"
					f" the response: {response}"
				)
				return response_list

		elif response_type == "definition":
			logger.warning("not implemented yet for definition")
			# TODO
			pass

	def build(self, construct: str) -> None:
		"""Union of tokens from each source. Remove tokens that were removed.
		:return: None
		"""
		# add
		tokens = []
		for source in self.constructs[construct]["tokens_metadata"]:
			if self.constructs[construct]["tokens_metadata"][source]["action"] in ["add", "create", "manually added"]:
				tokens_i = self.constructs[construct]["tokens_metadata"][source]["tokens"]
				tokens.extend(tokens_i)
		tokens = list(np.unique(tokens))
		self.constructs[construct]["tokens"] = tokens

		# remove
		remove = []
		for source in self.constructs[construct]["tokens_metadata"]:
			if self.constructs[construct]["tokens_metadata"][source]["action"] == "remove":
				tokens_i = self.constructs[construct]["tokens_metadata"][source]["tokens"]
				remove.extend(tokens_i)
		remove = list(np.unique(remove))
		self.constructs[construct]["remove"] = remove

		# If token is in override_remove, it will not be removed
		remove = [n for n in remove if n not in self.constructs[construct]["override_remove"]]

		remove_from_all_constructs = self.get_attribute("remove_from_all_constructs")
		if isinstance(remove_from_all_constructs, list) and len(remove_from_all_constructs) > 0:
			# if self.constructs['config'].get('remove_from_all_constructs')> 0
			# # if getattr(self, 'remove_from_all_constructs', []) > 0:
			self.constructs[construct]["remove"] = list(np.unique(remove + remove_from_all_constructs))

		# Remove tokens that are in remove list
		self.constructs[construct]["tokens"] = [n for n in tokens if n not in remove]

		return

	def add(
		self,
		construct: str,
		section: str = "tokens",
		value: Optional[Union[str, List[str]]] = None,  # str, list or 'create'
		domain: Optional[str] = None,
		examples: Optional[List[str]] = None,
		definition: Optional[str] = None,
		definition_references: Optional[str] = None,
		# Only used if value == 'create'. if value == 'create', do API request with LiteLLM
		prompt: Optional[str] = None,  # str, None will create default prompt
		source: Optional[
			str
		] = None,  # str: model such as 'command-nightly", see litellm for models, or description "manually added by DML". Cohere 'command-nightly' models offer 5 free API calls per minute.
		api_key: Optional[str] = None,
		temperature: float = 0.1,
		top_p: float = 1,
		seed: int = 42,
		timeout: int = 120,
		num_retries: int = 2,
		max_tokens: Optional[
			int
		] = 150,  # If set to None (whatever the model decides), it make take a long time and generate a lot of phrases combining the words which may be redundant. default is 150
		remove_parentheses_definition: bool = True,  # TODO: remove double spaces
		verbose: bool = True,  # True displays warnings
	) -> None:
		"""Add a construct or modify an existing one in the lexicon."""
		self.construct_names.append(construct)
		self.construct_names = list(set(self.construct_names))

		if construct not in self.constructs.keys():
			if verbose:
				logger.info(
					f"Adding '{construct}' to the lexicon (it was not previously there). If you meant to add to an existing construct"
					f" (and have a typo in the construct name, which is case sensitive), run: del your_lexicon_name.constructs['{construct}']"
				)
			self.constructs[construct] = {
				"prompt_name": construct.replace("_", " ").lower(),
				"variable_name": generate_variable_name(
					construct
				),  # you can replace it later with lexicon.add(construct, section='variable_name', value=new_name)
				"domain": domain,
				"examples": examples,
				"definition": definition,
				"definition_references": definition_references,
				"tokens": [],
				"tokens_lemmatized": [],
				"remove": [],
				"override_remove": [],
				"tokens_metadata": {},
			}
		else:
			for section_str, section_value in [
				("domain", domain),
				("examples", examples),
				("definition", definition),
				("definition_references", definition_references),
			]:
				if str(section_value) != "None":
					self.constructs[construct][section_str] = section_value

		ts = datetime.datetime.utcnow().strftime("%y-%m-%dT%H-%M-%S.%f")  # so you don't overwrite, and save timestamp

		if section == "tokens":
			if isinstance(value, str):
				if value == "create":
					if not isinstance(prompt, str):
						# generate default prompt
						prompt = generate_prompt(
							construct,
							prompt_name=self.constructs[construct]["prompt_name"],
							prompt="default",
							domain=self.constructs[construct]["domain"],
							# domain=self.constructs[construct]['domain'], # need to add domain to construct dict
							definition=self.constructs[construct]["definition"],
							examples=self.constructs[construct]["examples"],
							output_format="default",
							remove_parentheses_definition=remove_parentheses_definition,
						)
					# else, use prompt provided in arguments
					start = time.time()
					response = api_request(
						prompt,
						model=source,
						api_key=api_key,
						temperature=temperature,
						top_p=top_p,
						timeout=timeout,
						num_retries=num_retries,
						max_tokens=max_tokens,
						seed=seed,
					)
					end = time.time()
					time_elapsed = end - start
					time_elapsed = round(time_elapsed, 1)

					tokens = self.clean_response(response, response_type="tokens")

					tokens = list(np.unique(tokens))
					source_info = (
						f"{source}, temperature-{temperature}, top_p-{top_p}, max_tokens-{max_tokens}, seed-{seed},"
						f" {ts}"
					)
					self.constructs[construct]["tokens_metadata"][source_info] = {
						"action": "create",
						"tokens": tokens,
						"prompt": prompt,
						"time_elapsed": time_elapsed,
					}
				else:
					raise Exception(
						"value needs to be list of token/s. This operation will not add anything to lexicon. Returning"
						" None."
					)
					return
			elif isinstance(value, list):
				# manually add tokens
				tokens = value.copy()
				source_info = f"{source} {ts}"
				tokens = [n.strip() for n in tokens]
				self.constructs[construct]["tokens_metadata"][source_info] = {
					"action": "manually added",
					"tokens": tokens,
				}
			else:
				raise TypeError(
					"value needs to be list of token/s. This operation will not add anything to lexicon. Returning"
					" None."
				)
				return

			remove_tokens = self.constructs[construct]["remove"]
			try_to_add_but_removed = [n for n in tokens if n in remove_tokens]
			if len(try_to_add_but_removed) > 0:
				logger.warning(
					f"These tokens are trying to be added to the construct '{construct}' but are listed in the 'remove'"
					f" section of tokens_metadata: {try_to_add_but_removed}.\nThey will only be added to the"
					f" tokens_metadata not to the final tokens. You can override the previously removed tokens by"
					f" adding them to the 'override_remove': lexicon.add('{construct}', section='override_remove',"
					f" value=tokens_to_be_included). You can also delete any added or removed set of tokens from"
					f" metadata: \ndel lexicon.constructs['{construct}']['tokens_metadata']['source']\nfollowed"
					f" by:\nlexicon.build('{construct})\n"
				)

			# build (add all sources, remove from remove section
			self.build(construct)

		elif section in ["prompt_name", "definition", "definition_references", "examples"]:
			self.constructs[construct][section] = value  # replace value

		return

		# todo: compute percentage of final tokens that are from each source
		# TODO: add tests for all possibilities

	def remove(
		self,
		construct: str,
		source: Optional[str] = None,
		remove_tokens: Optional[List[str]] = None,
		remove_substrings: Optional[List[str]] = None,
	) -> None:
		"""Remove tokens from the lexicon construct."""
		# adds list of tokens to 'remove' section of 'tokens_metadata'.
		ts = datetime.datetime.utcnow().strftime("%y-%m-%dT%H-%M-%S.%f")  # so you don't overwrite, and save timestamp
		if isinstance(source, str):
			source_info = f"{source} {ts}"
		elif source is None:
			source_info = f"{ts}"

		self.constructs[construct]["tokens_metadata"][source_info] = {
			"action": "remove",
			"tokens": remove_tokens,
		}

		# TODO: adapt remove_substrings to new code: add in tokens_metada? It's currently not in build.
		if isinstance(remove_substrings, list):
			tokens = self.constructs[construct]["tokens"]
			p = re.compile("|".join(map(re.escape, remove_substrings)))  # escape to handle metachars
			tokens = [p.sub("", s).strip() for s in tokens]
			# Add examples
			for example in self.examples:
				if example not in tokens:
					tokens.insert(0, example)
			# Add construct
			# if construct not in tokens:
			# 	tokens.insert(0, construct)

			tokens = list(np.unique(tokens))
			self.constructs[construct]["tokens"] = tokens
		# self.constructs[construct]["tokens_removed"] =+ (tokens_len - len(tokens))
		self.build(construct)
		return

	def remove_tokens_containing_token(self, construct: str) -> None:
		"""Remove tokens from the specified construct that contain other tokens as substrings.

		Args:
				construct (str): The name of the construct from which tokens will be removed.

		Returns:
				None
		"""
		tokens = self.constructs[construct]["tokens"]
		tokens_len = len(tokens)
		partial_matching_strings, matched_substrings = find_partial_matching_strings(tokens, tokens)
		tokens = [n for n in tokens if n not in partial_matching_strings]  # remove tokens containing tokens
		self.constructs[construct]["tokens"] = tokens
		self.constructs[construct]["tokens_removed"] = +(tokens_len - len(tokens))
		return

	def to_pandas(
		self,
		add_ratings_columns: bool = True,
		add_metadata_rows: bool = True,
		order: Optional[List[str]] = None,
		tokens: str = "tokens",
	) -> pd.DataFrame:
		"""Convert the lexicon to a pandas DataFrame.

		Args:
				add_ratings_columns (bool, optional): Whether to add columns for ratings (`_include`, `_add`) for each construct. Defaults to True.
				add_metadata_rows (bool, optional): Whether to add metadata rows (e.g., definitions, examples) below each construct's column. Defaults to True.
				order (Optional[List[str]], optional): The order of constructs to use for the DataFrame columns. If None, the order of `self.constructs` keys is used. Defaults to None.
				tokens (str, optional): The key to use for the tokens in each construct. Defaults to "tokens".

		Returns:
				pd.DataFrame: A pandas DataFrame representing the lexicon.

		Example:
				lexicon: dictionary with at least
						{'construct 1': {'tokens': list of strings}
						}
		"""
		if order:
			warn_missing(self.constructs, order, output_format="pandas / csv")
		lexicon_df = []
		constructs = order.copy() if isinstance(order, list) else self.constructs.keys()
		for construct in constructs:
			df_i = pd.DataFrame(self.constructs[construct][tokens], columns=[construct])
			if add_ratings_columns:
				df_i[construct + "_include"] = [np.nan] * df_i.shape[0]
				df_i[construct + "_add"] = [np.nan] * df_i.shape[0]
			lexicon_df.append(df_i)

		lexicon_df = pd.concat(lexicon_df, axis=1)

		if add_metadata_rows:
			metadata_df_all = []
			if order is None:
				order = self.constructs.keys()

			for construct in order:
				# add definition, examples, prompt_name as rows below each construct's column
				definition = self.constructs[construct]["definition"]
				definition_references = self.constructs[construct]["definition_references"]
				examples = self.constructs[construct]["examples"]
				prompt_name = self.constructs[construct]["prompt_name"]
				metadata_df = pd.DataFrame(
					[prompt_name, definition, definition_references, examples], columns=[construct]
				)
				metadata_df[f"{construct}_include"] = [""] * len(metadata_df)
				metadata_df[f"{construct}_add"] = [""] * len(metadata_df)
				metadata_df_all.append(metadata_df)

			metadata_df_all = pd.concat(metadata_df_all, axis=1)
			lexicon_df = pd.concat([metadata_df_all, lexicon_df], axis=0, ignore_index=True)

			metadata_indeces = ["Prompt name", "Definition", "Reference", "Examples"]
			new_index = metadata_indeces + [
				n - len(metadata_indeces) for n in lexicon_df.index[len(metadata_indeces) :].tolist()
			]
			lexicon_df.index = new_index

		return lexicon_df

	def to_dict(self) -> Dict[str, List[str]]:
		"""Convert the lexicon to a dictionary."""

		lexicon_dict = {}
		for c in self.constructs:
			lexicon_dict[c] = self.constructs[c]["tokens"]
		return lexicon_dict

	def save(
		self,
		output_dir: str,
		filename: Optional[str] = None,
		output_format: Union[str, List[str]] = ("pickle", "json", "json_metadata", "csv", "csv_ratings"),
		order: Optional[List[str]] = None,
		timestamp: Union[bool, str] = True,
	) -> None:
		"""Save the lexicon to a file."""
		os.makedirs(output_dir, exist_ok=True)

		if filename is None:
			filename = self.name.replace(" ", "-").lower()
		path = output_dir + "/" + filename
		if timestamp:
			if isinstance(timestamp, str):
				path += path + f"_{timestamp}"
			elif timestamp is True:
				timestamp = generate_timestamp(format="%y-%m-%dT%H-%M-%S")
			path += f"_{timestamp}"

		if "pickle" in output_format:
			dill.dump(self, file=open(path + ".pickle", "wb"))  # save as object
		if "json" in output_format:
			save_json(self.constructs, path, with_metadata=True, order=order)
		if "json_metadata" in output_format:
			save_json(self.constructs, path, with_metadata=False, order=order)
		if "csv" in output_format:
			lexicon_df = self.to_pandas(add_ratings_columns=False, order=order)
			lexicon_df.to_csv(path + ".csv")
		if "csv_ratings" in output_format:
			lexicon_df = self.to_pandas(add_ratings_columns=True, order=order)
			lexicon_df.to_csv(path + "_ratings.csv")

		logger.info(f"Saved lexicon to {path}")

	def extract(
		self,
		docs: List[str],
		documents_df: Optional[pd.DataFrame] = None,
		normalize: bool = True,
		return_zero: List[str] = [],
		return_matches: bool = True,
		add_word_count: bool = True,
		add_lemmatized_lexicon: bool = True,
		lemmatize_docs: bool = False,
		exact_match_n: int = 4,
		exact_match_tokens: List[str] = [],
		save_dir: Union[str, bool] = False,
		save_append_to_filename: Optional[str] = None,
		save_as: str = "json",
	) -> Union[
		Tuple[pd.DataFrame, dict, dict, dict],
		pd.DataFrame,
	]:
		"""Extract features from documents by counting lexicon matches and optionally normalizing them.

		This method processes a list of documents by matching tokens from a lexicon against the documents.
		The matches can be normalized by word count to account for document length, making a match in a shorter
		document weigh more than a match in a longer document. The method can also return detailed information
		about the matches and save the results to disk.

		Args:
				docs (List[str]): List of documents to be processed.
				documents_df (Optional[pd.DataFrame], optional): DataFrame to which the output will be concatenated. Must have the same number of rows as the length of `docs`. Defaults to None.
				normalize (bool, optional): Whether to normalize the extracted features by word count. If True, matches in shorter documents will be weighed higher than in longer documents. Defaults to True.
				return_zero (List[str], optional): List of tokens for which to return a count of zero (e.g., if it is often producing errors). Defaults to [].
				return_matches (bool, optional): Whether to return detailed information about the matches. Defaults to True.
				add_word_count (bool, optional): Whether to add the word count of each document to the extracted features. Defaults to True.
				add_lemmatized_lexicon (bool, optional): Whether to include lemmatized tokens in the lexicon along with the original tokens. Defaults to True.
				lemmatize_docs (bool, optional): Whether to lemmatize the documents before processing. This increases the number of matches but may lead to less interpretable results. Defaults to False.
				exact_match_n (int, optional): The maximum length for exact matches. If a lexicon token is short, it would a substring of many document words, creating false positives. Tokens shorter than or equal to this length will be treated as exact matches to avoid false positives. Defaults to 4.
				exact_match_tokens (List[str], optional): List of tokens to be treated as exact matches. Defaults to [].
				save_dir (Union[str, bool], optional): Directory to save the extracted features. If `False`, results are not saved. Defaults to False.
				save_append_to_filename (Optional[str], optional): String to append to the filename when saving results. Defaults to None.
				save_as (str, optional): Format in which to save the output. Can be "json" or "pickle". Defaults to 'json'.

		Returns:
				Union[
						Tuple[pd.DataFrame, dict, dict, dict],
						pd.DataFrame,
				]:
				- If `return_matches` is True:
						Tuple containing:
						- feature_vectors_df (pd.DataFrame): DataFrame containing the extracted features.
						- matches_by_construct (dict): Dictionary with matched tokens by construct.
						- matches_doc2construct (dict): Dictionary mapping document IDs to matched tokens for each construct.
						- matches_construct2doc (dict): Dictionary mapping constructs to matched tokens for each document.
				- If `return_matches` is False:
						- feature_vectors_df (pd.DataFrame): DataFrame containing the extracted features.

		Example:
				>>> features, matches = lexicons.extract(docs, lexicons_d, normalize=True)
		"""
		if isinstance(documents_df, pd.DataFrame):
			try:
				assert documents_df.shape[0] == len(docs)
			except Exception as e:
				raise ValueError(
					f"Error {e}: documents_df should have the same amount of rows as the length of documents"
				)

		lexicon_dict = (
			self.constructs.copy()
		)  # keys are constructs, values are dictionaries with tokens, definitions and other metadata.
		docs = [doc.replace("\n", " ").replace("  ", " ").replace("“", "").replace("”", "") for doc in docs]
		if lemmatize_docs:
			print("lemmatizing docs...")
			docs = lemmatizer.spacy_lemmatizer(docs, language="en")  # custom function
			docs = [" ".join(n) for n in docs]

		print("extracting... ")
		docs2 = docs.copy()
		docs = []
		for doc in docs2:
			if "ness" in doc:
				# TODO: should be optional
				# No words really start with ness
				# if ' ness' in doc or doc.lower().startswith('ness'):
				# 	# eg., nec
				# 	continue
				# else:
				ness_tokens = [
					word.strip(" ,.!?*%)]|>#") for word in doc.split(" ") if word.strip(" ,.!?*%)]|>#").endswith("ness")
				]  # ['sadness,', 'loneliness,']

				tokens_adj = [
					word.replace("iness", "y").replace("ness", "") for word in ness_tokens
				]  # ['sad,', 'lonely']
				for token_ness, token_adj in zip(ness_tokens, tokens_adj):
					doc = doc.replace(token_ness, f"{token_ness} [{token_adj}]")
			docs.append(doc)

		feature_vectors_df = {}
		matches = {}
		matches_construct2doc = {}
		matches_doc2construct = {}
		for i in range(len(docs)):
			matches_doc2construct[i] = {}

		for construct in tqdm(list(lexicon_dict.keys()), position=0):
			# if lemmatize_lexicon:

			# 	lexicon_tokens = lemmatizer.spacy_lemmatizer(lexicon_tokens, language='en') # custom function
			# 	lexicon_tokens = [' '.join(n) for n in lexicon_tokens]

			if add_lemmatized_lexicon:
				# replace lexicon_tokens with lemmatized tokens

				lexicon_tokens = lexicon_dict.get(construct)["tokens_lemmatized"]
				if lexicon_tokens == []:
					# Lemmatize
					logger.warning(
						"Lemmatizing the lexicon tokens. We recommend you lemmatize before extracting (my_lexicon = lemmatize_tokens(my_lexicon)) so you can save time if you"
						" want to repeat extraction on different documents."
					)
					lexicon_tokens = lexicon_dict.get(construct)["tokens"]
					# If you add lemmatized and nonlemmatized you'll get double count in many cases ("plans" in doc will be matched by "plan" and "plans" in lexicon)

					lexicon_tokens_lemmatized = lemmatizer.spacy_lemmatizer(
						lexicon_tokens, language="en"
					)  # custom function
					lexicon_tokens_lemmatized = [" ".join(n) for n in lexicon_tokens_lemmatized]
					lexicon_tokens += lexicon_tokens_lemmatized
					lexicon_tokens = list(np.unique(lexicon_tokens))  # unique set
					# save
					lexicon_dict.get(construct)["tokens_lemmatized"] = lexicon_tokens

					# Save
					self.constructs[construct]["tokens_lemmatized"] = lexicon_tokens

			else:
				lexicon_tokens = lexicon_dict.get(construct)["tokens"]

				"""
				lemmatizer.spacy_lemmatizer(['distressed'])
				'distressed' > "distress", only "distress" is kept, to avoid counting twice.
				"drained" > "drain", only "drain" is kept unless its in the except_exact_match list"
				"die" and "died" will be kep because they are in the exact match list, because <= exact_match_n
				"catastrophizing" > "catastrophize", both are kept
				'forced to exist' > 'force to exist'
				"I'm a drag"> "I am a drag", both will be kept, because one is not a substring of the other
				"grabbed me"> "grab me", both will be kept, because one is not a substring of the other
				"""
			# remove tokens that contain tokens to avoid counting twice
			# except for exact_match_n and exact matches.
			except_exact_match = list(
				np.unique(exact_match_tokens + [n for n in lexicon_tokens if len(n) <= exact_match_n])
			)  # TODO: maybe "died">"die"
			lexicon_tokens = remove_tokens_containing_token(lexicon_tokens, except_exact_match=except_exact_match)

			if return_matches:
				counts_and_matched_tokens = [
					count_lexicons_in_doc(
						doc,
						tokens=lexicon_tokens,
						return_zero=return_zero,
						return_matches=return_matches,
						exact_match_n=exact_match_n,
						exact_match_tokens=exact_match_tokens,
					)
					for doc in docs
				]
				matches_construct2doc[construct] = counts_and_matched_tokens
				# for a single construct
				for i, doc_i in enumerate(counts_and_matched_tokens):
					# each document for that construct
					matches_doc2construct[i][construct] = doc_i

				counts = [n[0] for n in counts_and_matched_tokens]
				matched_tokens = [n[1] for n in counts_and_matched_tokens if n[1] != []]
				matches[construct] = matched_tokens

			else:
				counts = [
					count_lexicons_in_doc(
						doc,
						tokens=lexicon_tokens,
						return_zero=return_zero,
						return_matches=return_matches,
						exact_match_n=exact_match_n,
						exact_match_tokens=exact_match_tokens,
					)
					for doc in docs
				]
			# one_construct = one_construct/word_counts #normalize

			feature_vectors_df[construct] = counts

		# # feature_vector = extract_NLP_features(post, features) #removed feature_names from output
		# if len(feature_vector) != 0:
		#     raw_series = list(df_subreddit.iloc[pi])
		#     subreddit_features = subreddit_features.append(pd.Series(raw_series + feature_vector, index=full_column_names), ignore_index=True)

		# feature_vectors_df0   = pd.DataFrame(docs, columns = ['docs'])
		# feature_vectors_df = pd.concat([feature_vectors_df0,pd.DataFrame(feature_vectors_df)],axis=1)
		feature_vectors_df = pd.DataFrame(feature_vectors_df)

		#     feature_vectors_df   = pd.DataFrame(docs)
		#     feature_vectors_df['docs']=docs

		if normalize:
			wc = word_count(docs, return_zero=return_zero)
			wc = np.array(wc)
			feature_vectors_df_normalized = np.divide(feature_vectors_df.values.T, wc).T
			feature_vectors_df = pd.DataFrame(
				feature_vectors_df_normalized, index=feature_vectors_df.index, columns=feature_vectors_df.columns
			)

		if add_word_count and normalize:
			feature_vectors_df["word_count"] = wc
		elif add_word_count and not normalize:
			wc = word_count(docs, return_zero=return_zero)
			feature_vectors_df["word_count"] = wc

		# feature_vectors_df = feature_vectors_df/wc

		# add column with documents

		feature_vectors_df.insert(0, "document", docs)
		feature_vectors_df.insert(0, "document_id", range(len(docs)))

		if save_dir:
			lexicon_name_clean = generate_variable_name(str(self.name) + "_v" + str(self.version))
			if save_dir[-1] == "/":
				save_dir = (
					save_dir
					+ f"{lexicon_name_clean}_counts_and_matches_"
					+ generate_timestamp(format="%y-%m-%dT%H-%M-%S")
					+ "/"
				)
			else:
				save_dir = (
					save_dir
					+ f"/{lexicon_name_clean}_counts_and_matches_"
					+ generate_timestamp(format="%y-%m-%dT%H-%M-%S")
					+ "/"
				)
			os.makedirs(save_dir, exist_ok=True)

		if return_matches:
			# all lexicons
			matches_by_construct = {}
			for lexicon_name_i in list(lexicon_dict.keys()):
				if matches.get(lexicon_name_i):
					x = Counter([n for i in matches.get(lexicon_name_i) for n in i])
					matches_by_construct[lexicon_name_i] = {
						k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)
					}
			# Counter([n for i in matches.get(lexicon_name_i) for n in i]) for lexicon_name_i in lexicon_dict.keys()]

			if isinstance(documents_df, pd.DataFrame):
				documents_df.reset_index(drop=True, inplace=True)
				feature_vectors_df = pd.concat([documents_df, feature_vectors_df], axis=1)

			if save_dir:
				if save_append_to_filename:
					save_append_to_filename = "_" + save_append_to_filename
				else:
					save_append_to_filename = ""
				filename = lexicon_name_clean + f"_counts{save_append_to_filename}.csv"
				feature_vectors_df.to_csv(save_dir + filename, index=False)
				# write a txt with arguments
				frame = inspect.currentframe()
				args, _, _, values = inspect.getargvalues(frame)
				arguments = {arg: values[arg] for arg in args}

				exclude = ["documents", "documents_df", "cosine_scores_docs_all", "docs"]
				with open(save_dir + f"arguments_log{save_append_to_filename}.txt", "w") as file:
					for name, value in arguments.items():
						if name not in exclude:
							file.write(f"{name}: {value}\n")

				if save_as == "pickle":
					# more lightweight
					pickle.dump(
						matches_by_construct,
						open(
							save_dir + lexicon_name_clean + f"_matches_by_construct{save_append_to_filename}.pickle",
							"wb",
						),
					)
					pickle.dump(
						matches_doc2construct,
						open(
							save_dir + lexicon_name_clean + f"_matches_doc2construct{save_append_to_filename}.pickle",
							"wb",
						),
					)
					pickle.dump(
						matches_construct2doc,
						open(
							save_dir + lexicon_name_clean + f"_matches_construct2doc{save_append_to_filename}.pickle",
							"wb",
						),
					)
				elif save_as == "json":
					with open(
						save_dir + lexicon_name_clean + f"_matches_by_construct{save_append_to_filename}.json", "w"
					) as json_file:
						json.dump(matches_by_construct, json_file, indent=4)
					with open(
						save_dir + lexicon_name_clean + f"_matches_doc2construct{save_append_to_filename}.json", "w"
					) as json_file:
						json.dump(matches_doc2construct, json_file, indent=4)
					with open(
						save_dir + lexicon_name_clean + f"_matches_construct2doc{save_append_to_filename}.json", "w"
					) as json_file:
						json.dump(matches_construct2doc, json_file, indent=4)

			return feature_vectors_df, matches_by_construct, matches_doc2construct, matches_construct2doc
		else:
			if save_dir:
				if save_append_to_filename:
					save_append_to_filename = "_" + save_append_to_filename
				else:
					save_append_to_filename = ""
				filename = lexicon_name_clean + f"_counts{save_append_to_filename}.csv"
				feature_vectors_df.to_csv(save_dir + filename, index=False)
				# write a txt with arguments
				frame = inspect.currentframe()
				args, _, _, values = inspect.getargvalues(frame)
				arguments = {arg: values[arg] for arg in args}
				exclude = ["documents", "documents_df", "docs"]
				with open(save_dir + f"arguments_log{save_append_to_filename}.txt", "w") as file:
					for name, value in arguments.items():
						if name not in exclude:
							file.write(f"{name}: {value}\n")
			return feature_vectors_df


def generate_timestamp(format: str = "%y-%m-%dT%H-%M-%S-%f") -> str:
	"""
	Generate a timestamp in a specific format.

	Args:
			format (str, optional): The format in which to generate the timestamp. Defaults to "%y-%m-%dT%H-%M-%S-%f".

	Returns:
			str: The generated timestamp as a string.
	"""
	ts = datetime.datetime.utcnow().strftime(format)  # Generate the timestamp in UTC format
	return ts


def load_lexicon(name: Optional[str] = None, path: Optional[str] = None) -> "Lexicon":
	"""Load a lexicon from a file."""
	script_dir = os.path.dirname(__file__)  # Directory of the script being run

	if name == "srl_v1-0":
		path = os.path.join(
			script_dir,
			"data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_validated_24-08-02T21-27-35.pickle",
		)
	elif name == "srl_prototypes_v1-0":
		path = os.path.join(
			script_dir,
			"data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_validated_prototypical_tokens_24-08-07T16-25-19.pickle",
		)
	else:
		# Assuming the user provided a path, it should be an absolute path or relative to script_dir
		path = os.path.join(script_dir, path) if path else None
	prior_lexicon = dill.load(open(path, "rb"))
	for c in prior_lexicon.constructs:
		tokens = prior_lexicon.constructs[c]["tokens"]
		tokens_str = []
		for token in tokens:
			if type(token) == np.str_:
				token = token.item()
				tokens_str.append(token)
			else:
				tokens_str.append(token)
		prior_lexicon.constructs[c]["tokens"] = tokens

	return prior_lexicon


def dict_to_lexicon(lexicon_dict: Dict[str, List[str]]) -> Lexicon:
	"""Convert a dictionary to a Lexicon object."""
	my_lexicon = Lexicon()  # Initialize lexicon

	for c in lexicon_dict.keys():
		my_lexicon.add(c, section="tokens", value=lexicon_dict[c])
	return my_lexicon


def load_json_to_lexicon(json_path: str) -> Lexicon:
	"""Load a lexicon from a JSON file."""
	# Final tokens:
	with open(json_path, "r") as f:
		data = json.load(f)

	my_lexicon = Lexicon()  # Initialize lexicon

	for c in data.keys():
		for section in data[c].keys():
			my_lexicon.add(c, section=section, value=data[c][section])
	return my_lexicon

	# Copy full lexicon but remove the ones that aren't
	# ============================================================================
	# import json
	# import copy

	# json_path = './../src/construct_tracker/data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_validated_prototypical_tokens_24-03-06T00-47-30.json'
	# with open(json_path, 'r') as f:
	# 	data = json.load(f)

	# srl_prototypes = copy.deepcopy(srl)         # Initialize lexicon

	# for c in srl_prototypes.constructs.keys():
	# 	prototypes_c = set(data[c]['tokens']) #==3 [0-3]
	# 	all_validated_tokens = set(srl.constructs[c]['tokens']) # > 1.3 [0-3]
	# 	removed = list(all_validated_tokens - prototypes_c)
	# 	# srl_prototypes.constructs[c]['tokens'] = data[c]['tokens']
	# 	srl_prototypes.remove(c, remove_tokens = removed, source = 'Prototypes with an average of less than 3/3 across raters' ) #redu

	# If loading some info from one version, and other info from another version:
	# ============================================================================
	# # # Metadata:
	# with open('./../src/construct_tracker/data/lexicons/suicide_risk_lexicon_v1-0/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_calibrated_matched_tokens_unvalidated_24-02-15T22-12-18.json', 'r') as f:
	# 	metadata = json.load(f)

	# srl = Lexicon()         # Initialize lexicon
	# for c in data.keys():
	# 	for section in data[c].keys():
	# 		if section in ['tokens', 'tokens_lemmatized']:
	# 			srl.add(c, section = section, value = data[c][section])
	# 		srl.add(c, section = section, value = metadata[c][section])
	# return srl

	# Then you need to add other info and save.
	# ============================================================================
	# srl.name = 'Suicide Risk Lexicon'		# Set lexicon name
	# srl.description = '49 risk factors for suicidal thoughts and behaviors plus one construct about kinship, validated by clinical experts. If you use, please cite publication.'
	# srl.creator = 'Daniel M. Low (Harvard University)' 				# your name or initials for transparency in logging who made changes
	# srl.version = '1.0'				# Set version. Over time, others may modify your lexicon, so good to keep track. MAJOR.MINOR. (e.g., MAJOR: new constructs or big changes to a construct, Minor: small changes to a construct)
	# srl.save('./../src/construct_tracker/data/lexicons/suicide_risk_lexicon_v1-0/', filename = 'suicide_risk_lexicon_validated')


def warn_missing(dictionary: Dict[str, Any], order: List[str], output_format: Optional[str] = None) -> None:
	"""Warn if there are missing constructs."""
	missing = [n for n in dictionary if n not in order]
	if len(missing) > 0:
		logger.warning(
			f"These constructs were NOT SAVED in {output_format} because were not in order argument: {missing}. They"
			" are saved in the lexicon pickle file"
		)
	return


def save_json(
	dictionary: Dict[str, Any], path: str, with_metadata: bool = True, order: Optional[List[str]] = None
) -> None:
	"""Save the lexicon as a JSON file."""
	if order:
		warn_missing(dictionary, order, output_format="json")
		dictionary = {k: dictionary[k] for k in order}

	if with_metadata:
		with open(path + "_metadata.json", "w") as fp:
			json.dump(dictionary, fp, indent=4)
	else:
		dictionary_wo_metadata = {}
		for construct in dictionary:
			dictionary_wo_metadata[construct] = dictionary[construct].copy()
			dictionary_wo_metadata[construct]["tokens_metadata"] = "see metadata file for tokens and sources"
			dictionary_wo_metadata[construct]["remove"] = (
				"see metadata file for tokens that were removed through human ratings/coding"
			)
		# for source in dictionary_wo_metadata[construct]['tokens_metadata'].keys():
		# 	del dictionary_wo_metadata[construct]['tokens_metadata'][source]['tokens']

		with open(path + ".json", "w") as fp:
			json.dump(dictionary_wo_metadata, fp, indent=4)
	return


def count_lexicons_in_doc(
	doc: str,
	tokens: Optional[List[str]] = None,
	return_zero: Optional[List[str]] = None,
	return_matches: bool = False,
	exact_match_n: int = 4,
	exact_match_tokens: Optional[List[str]] = None,
) -> Union[int, Tuple[int, List[str]]]:
	"""Count occurrences of lexicon tokens in a document. Helps check for false positives.

	Args:
					doc: The document text.
					tokens: List of lexicon tokens.
					return_zero: Optional; tokens to return zero for.
					return_matches: Optional; whether to return matched tokens.
					exact_match_n: Optional; maximum length of tokens for exact matches.
					exact_match_tokens: Optional; list of tokens for exact matches.

	Returns:
					The count of matches and optionally the matched tokens.
	"""
	# remove punctuation except apostrophes because we need to search for things like "don't want to live"

	doc = doc.lower().replace("-", " ")
	table = str.maketrans("", "", string.punctuation.replace("'", ""))  # Apostrophe is preserved
	text = doc.translate(table)

	# text = re.sub("[^\\w\\d'\\s]+", "", doc.lower())

	counter = 0
	matched_tokens = []
	for token in tokens:
		token = token.lower()
		# if isinstance(starts_with, list):
		# 	TODO: include not only if exact match, but also if starts with
		if len(token) <= exact_match_n or token in exact_match_tokens:
			matches = len([n for n in text.split(" ") if n == token])
		else:
			matches = text.count(token)
		counter += matches
		if return_matches and matches > 0:
			# print(matches, token)
			matched_tokens.append(token)

	if return_matches:
		return counter, matched_tokens
	else:
		return counter


# def count_lexicons_in_doc(
# 	doc: str,
# 	tokens: Optional[List[str]] = None,
# 	return_zero: Optional[List[str]] = None,
# 	return_matches: bool = False,
# ) -> Union[int, Tuple[int, List[str]]]:
# 	"""Count occurrences of lexicon tokens in a document.

# 	Args:
# 			doc: The document text.
# 			tokens: List of lexicon tokens.
# 			return_zero: Optional; tokens to return zero for.
# 			return_matches: Optional; whether to return matched tokens.

# 	Returns:
# 			The count of matches and optionally the matched tokens.
# 	"""
# 	tokens = tokens or []
# 	return_zero = return_zero or []

# 	text = re.sub("[^\\w\\d'\\s]+", "", doc.lower())
# 	counter = 0
# 	matched_tokens = []
# 	for token in tokens:
# 		token = token.lower()
# 		matches = text.count(token)
# 		counter += matches
# 		if return_matches and matches > 0:
# 			matched_tokens.append(token)
# 	if return_matches:
# 		return counter, matched_tokens
# 	else:
# 		return counter


def remove_tokens_containing_token(tokens: List[str], except_exact_match: Optional[List[str]] = None) -> List[str]:
	"""Remove tokens that contain other tokens, except for exact matches.


	Args:
					tokens: List of tokens to filter.
					except_exact_match: Optional; list of tokens to exclude from filtering.

	Returns:
					The filtered list of tokens.

	Example:
			remove_tokens_containing_token(['mourn', 'mourning', 'beat', 'beating'],  except_exact_match = ['beat'])
			# Should keep beat and beating because beat is in except_exact_match, so it won't be redundant if you search for both.
	"""
	if len(except_exact_match) > 0:
		tokens_substrings = [n for n in tokens if n not in except_exact_match]
		partial_matching_strings, matched_substrings = find_partial_matching_strings(tokens, tokens_substrings)
		# partial matching strings contain other strings, so they are redundant
	else:
		partial_matching_strings, matched_substrings = find_partial_matching_strings(tokens, tokens)
		# partial matching strings contain other strings, so they are redundant
	tokens = [n for n in tokens if n not in partial_matching_strings]  # remove tokens containing tokens
	return tokens


def find_match(s: str, pat: str) -> List[str]:
	"""
	Find matches of a pattern within a string.

	Args:
			s (str): The string to search within.
			pat (str): The pattern to search for.

	Returns:
			List[str]: A list of matching substrings.
	"""
	pat = r"(\w*%s\w*)" % pat  # Not thrilled about this line
	match = re.findall(pat, s)
	return match


def startswith_str(doc: str, pat: str) -> List[str]:
	"""
	Find tokens in a document that start with a specific pattern.

	Args:
			doc (str): The document to search within.
			pat (str): The pattern to match at the start of tokens.

	Returns:
			List[str]: A list of tokens that start with the pattern.
	"""
	tokens = doc.split(" ")
	match = [n for n in tokens if n.startswith(pat)]
	return match


def match(docs: List[str], token: str) -> List[str]:
	"""
	Find matches for a token within a list of documents.

	Args:
			docs (List[str]): A list of documents to search within.
			token (str): The token to search for.

	Returns:
			List[str]: A list of unique matches.
	"""
	matches = [find_match(doc.lower(), token) for doc in docs]
	# TODO: don't lower acronyms
	# matches = [startswith_str(doc.lower(), token) for doc in docs]
	matches = list(np.unique([n for i in matches for n in i]))
	return matches


def lemmatize_tokens(lexicon_object: object) -> object:
	"""
	Lemmatize tokens in a lexicon object.

	Args:
			lexicon_object (object): The lexicon object containing constructs and tokens.

	Returns:
			object: The updated lexicon object with lemmatized tokens added.
	"""
	# TODO: do not lemmatize "I", hit me > hit I, cut my > cut I, same with her and him > block he, block her.
	# UNLESS you lemmatize the doc as well.
	for c in tqdm(lexicon_object.constructs.keys(), position=0):
		srl_tokens = lexicon_object.constructs[c]["tokens"].copy()
		# If you add lemmatized and nonlemmatized you'll get double count in many cases ("plans" in doc will be matched by "plan" and "plans" in srl)
		srl_tokens_lemmatized = lemmatizer.spacy_lemmatizer(srl_tokens, language="en")  # custom function
		srl_tokens_lemmatized = [" ".join(n) for n in srl_tokens_lemmatized]
		srl_tokens += srl_tokens_lemmatized
		if lexicon_object.constructs[c]["remove"] is None:
			lexicon_object.constructs[c]["remove"] = []
		srl_tokens = [
			n.replace(" - ", "-").strip() for n in srl_tokens if n not in lexicon_object.constructs[c]["remove"]
		]
		srl_tokens = [
			n.replace("-", " ").strip() for n in srl_tokens if n not in lexicon_object.constructs[c]["remove"]
		]
		srl_tokens = list(np.unique(srl_tokens))  # unique set

		# TODO: add back in override_remove

		lexicon_object.constructs[c]["tokens_lemmatized"] = srl_tokens
	return lexicon_object


def add_remove_from_ratings_file(
	lexicon_df: pd.DataFrame, my_lexicon: object, remove_below_or_equal_to: int = 0
) -> object:
	"""
	Add or remove tokens from a lexicon based on a ratings file.

	Args:
			lexicon_df (pd.DataFrame): DataFrame containing the lexicon and ratings information.
			my_lexicon (object): The lexicon object to update.
			remove_below_or_equal_to (int, optional): Threshold for removing tokens based on ratings. Defaults to 0.

	Returns:
			object: The updated lexicon object.
	"""
	constructs = [n for n in lexicon_df.columns if "_" not in n]
	for construct in constructs:
		print(construct)
		# Add
		add_i = lexicon_df[~lexicon_df[construct + "_add"].isna()][construct + "_add"].tolist()
		if len(add_i) > 0:
			my_lexicon.add(construct, section="tokens", value=add_i, source=my_lexicon.creator)
		print("added:", add_i)

		# Remove
		remove_i = lexicon_df[lexicon_df[construct + "_include"] <= remove_below_or_equal_to][construct].tolist()
		if len(remove_i) > 0:
			my_lexicon.remove(construct, remove_tokens=remove_i, source=my_lexicon.creator)
		print("removed:", add_i)
		print()
	return my_lexicon


def display_highlighted_documents(highlighted_documents: List[str]) -> None:
	"""
	Display highlighted documents with HTML formatting.

	Args:
			highlighted_documents (List[str]): List of HTML formatted strings to display.

	Returns:
			None
	"""
	for doc in highlighted_documents:
		display(HTML(doc))
	return


def highlight_matches(
	documents: List[str],
	construct: str,
	matches_construct2doc: Dict[str, List[Tuple[int, List[str]]]],
	max_matches: int = 3,
	shuffle: bool = True,
	random_seed: int = 42,
) -> List[str]:
	"""Highlight the matches of a given construct in the provided documents.
	This function takes a list of documents and a construct to search for matches in the documents. It uses the `matches_construct2doc` dictionary to retrieve the matches for the given construct. The function then iterates over the matches and highlights the matched words in the corresponding document by replacing them with HTML tags. The highlighted documents are returned as a list.


	Args:
					documents: The list of documents to search for matches.
					construct: The construct to search for matches.
					matches_construct2doc: A dictionary mapping constructs to their corresponding matches in the documents.
					max_matches: The maximum number of matches to highlight, if available. Default is 3.
					shuffle: Whether to shuffle the documents before highlighting. Default is True.
					random_seed: Seed for the random number generator used in shuffling. Default is 42.

	Returns:
					A list of highlighted documents.

	Example:
					construct = "Compassion"

					N = 2

					documents = ['He is too competitive','Every time I speak with my cousin Bob, I have great moments of insight, clarity, and wisdom',"He meditates a lot, but he's not super smart"]
					matches_construct2doc = {'Insight': [(0, []), (2, ['clarity', 'wisdom']), (0, [])],
																													'Mindfulness': [(0, []), (2, ['clarity', 'insight']), (1, ['meditate'])],
																													'Compassion': [(0, []), (0, []), (0, [])]}
					highlighted_docs = highlight_matches(documents, construct,N, matches_construct2doc)
					display_highlighted_documents(highlighted_docs)
					["This is a <mark>test</mark> document.", "Another document.", "This document contains the <mark>construct</mark>."]
	"""

	highlighted_documents = []

	matches = matches_construct2doc.get(construct, [])
	if shuffle:
		random.seed(random_seed)
		# joint shuffle matches and documents

		combined = list(zip(documents, matches))
		random.shuffle(combined)
		documents, matches = zip(*combined)

	highlighted_docs_found = 0
	for doc_index, (matches_n, words) in enumerate(matches):
		if highlighted_docs_found >= matches_n:
			break
		if matches_n == 0 and not words:
			continue

		document = documents[doc_index]
		for word in words:
			document = document.replace(word, f"<mark>{word}</mark>")
		highlighted_documents.append(document)
		highlighted_docs_found += 1

	display_highlighted_documents(highlighted_documents)
	# return highlighted_documents


def avg_above_thresh(
	ratings: Dict[str, Dict[str, List[float]]], thresh: float = 1.3
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
	"""
	Calculate the average ratings for tokens and filter based on a threshold.

	Args:
			ratings (Dict[str, Dict[str, List[float]]]):
					Dictionary containing constructs and their tokens with associated ratings.
			thresh (float, optional):
					The threshold above which a token's average rating must be to be included. Defaults to 1.3.

	Returns:
			Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
					- ratings_avg2: Dictionary of constructs with lists of tokens that have an average rating above the threshold.
					- ratings_removed2: Dictionary of constructs with lists of tokens that were removed because their average rating was below the threshold or contained a zero.

	Examples:
			ratings = {
					construct_1: {token_1: [rating_1, rating_2, ...], token_2: [rating_1, rating_2, ...], ...},
					construct_2: {token_1: [rating_1, rating_2, ...], token_2: [rating_1, rating_2, ...], ...},
					...
			}
			ratings_avg, ratings_removed = avg_above_thresh(ratings, thresh=1.3)
			ratings_avg_prototypical, ratings_removed_prototypical = avg_above_thresh(ratings, thresh=3.0)
	"""
	ratings_avg = {}
	ratings_removed = {}

	for construct in ratings.keys():
		ratings_avg[construct] = []
		ratings_removed[construct] = []
		for token in ratings[construct].keys():
			avg_score = np.mean(ratings[construct][token])
			# var = np.var(ratings[construct][token])
			if 0 not in ratings[construct][token] and avg_score >= thresh:
				ratings_avg[construct].append(token)
			else:
				ratings_removed[construct].append(token)

	# Remove '_include' from dict keys
	ratings_avg2 = {}
	ratings_removed2 = {}

	for construct in ratings_avg.keys():
		ratings_avg2[construct.replace("_include", "")] = ratings_avg[construct]
		ratings_removed2[construct.replace("_include", "")] = ratings_removed[construct]

	return ratings_avg2.copy(), ratings_removed2.copy()


def merge_rating_dfs(
	ratings_dir: str, rating_files: List[str], constructs: List[str], excel_sheet_name: Optional[str] = None
) -> Dict[str, Dict[str, List[float]]]:
	"""
	Merge ratings from multiple files into a single dictionary structure.

	Args:
			ratings_dir (str):
					Directory containing the rating files.
			rating_files (List[str]):
					List of filenames containing ratings data.
			constructs (List[str]):
					List of constructs to be merged.
			excel_sheet_name (Optional[str], optional):
					Name of the Excel sheet to read if the file is in Excel format. Defaults to None.

	Returns:
			Dict[str, Dict[str, List[float]]]:
					A dictionary where each construct maps to another dictionary of tokens and their associated ratings.
					# {construct_1: {token_1: [rating_1, rating_2, ...], token_2: [rating_1, rating_2, ...], ...},
					# construct_2: {token_1: [rating_1, rating_2, ...], token_2: [rating_1, rating_2, ...], ...},
					# ...}
	"""
	
	all_ratings_per_construct = dict(zip(constructs, [{}] * len(constructs)))

	for file in rating_files:
		rater_i = None
		try:
			if file.endswith(".csv"):
				rater_i = pd.read_csv(ratings_dir + file)
			elif file.endswith(".xlsx"):
				sheet_name = excel_sheet_name if excel_sheet_name else file
				rater_i = pd.read_excel(ratings_dir + file, engine="openpyxl", sheet_name=sheet_name)
		except Exception as e:
			print(f"Failed to read {file}. Save as csv and try again. Error: {e}")
			continue  # Skip processing this file if it fails

		if rater_i is not None:  # Only process if the file is read successfully
			all_ratings_construct_i = {}
			for construct in constructs:
				all_ratings_construct_i[construct] = []

				rater_i_construct_j_df = rater_i[[construct, construct + "_add"]].dropna().values
				rater_i_construct_j_dict = dict(zip(rater_i_construct_j_df[:, 0], rater_i_construct_j_df[:, 1]))

				for token, rating in rater_i_construct_j_dict.items():
					if token not in all_ratings_per_construct[construct]:
						all_ratings_per_construct[construct][token] = []
					if str(rating) not in ["nan", "None", "", "NaN"]:
						all_ratings_per_construct[construct][token].append(float(rating))

	return all_ratings_per_construct
