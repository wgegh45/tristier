import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
import sys
from pathlib import Path
import unicodedata
import math

# ==================== 設定エリア ====================
# 比較するモデルのリスト（使用するモデルのコメントを外す）
COMPARE_MODELS = [
    "hfl_chinese-macbert-base",
    "hfl_chinese-macbert-large",
    "hfl_chinese-bert-wwm-ext",
    "hfl_chinese-roberta-wwm-ext",
    "hfl_chinese-electra-180g-base-generator",
    "hfl_chinese-electra-180g-large-generator",
### "ckiplab_bert-base-chinese",
    "ethanyt_guwenbert-base",
    "bert-base-chinese"
]

# モデル名の略称
MODEL_ABBREVIATIONS = {
    "hfl_chinese-macbert-base": "M-b",    # MacBERT Base（推奨）- 中国語マスク言語モデル
    "hfl_chinese-macbert-large": "M-l",   # MacBERT Large - 高精度版
    "hfl_chinese-bert-wwm-ext": "B-w",    # BERT WWM (Whole Word Masking)
    "hfl_chinese-roberta-wwm-ext": "R-w", # RoBERTa WWM Base
    "hfl_chinese-electra-180g-base-generator": "E-b",  # ELECTRA Generator Base
    "hfl_chinese-electra-180g-large-generator": "E-l", # ELECTRA Generator Large
### "ckiplab_bert-base-chinese": "C-b",   # CKIP BERT - 台湾中国語対応  GPL-3.0 licenseのため利用不可
    "ethanyt_guwenbert-base": "G-b",      # GuwenBERT - 古文中国語専用
    "bert-base-chinese": "B-b"            # BERT Base Chinese - Google公式
}

# 候補数
TOP_K = 10

# 文脈考慮モード（True: 行全体を考慮, False: 1文のみ考慮）
USE_FULL_LINE_CONTEXT = True

# 表示モード（"vertical": 縦並び, "horizontal": 横並び）
#DISPLAY_MODE = "vertical"

# 確率表示（True: 表示, False: 非表示）
SHOW_PROBABILITY = True

# 確率の小数点以下桁数
PROB_DECIMALS_DISPLAY = 4  # 画面表示用
PROB_DECIMALS_FILE = 6     # ファイル出力用

# モデルの最大トークン数
MAX_TOKENS = 512

# ==================== アンサンブルスコアリング設定 ====================
MIN_PROBABILITY_THRESHOLD = 0.015  # 1.5%未満は除外
MAX_RANK_TO_CONSIDER = 10  # 評価する最大順位
MIN_SUPPORT = 2  # ★追加: 最小支持モデル数

# 使用するスコア計算方式
RANK_SCORE_METHOD = 'exponential'  # 'exponential', 'logarithmic', 'linear', 'inverse'

# アンサンブル結果の表示候補数
ENSEMBLE_TOP_N = 10

# 使用する方式の選択
ENSEMBLE_METHODS = ["rank", "probability", "hybrid"]
# =====================================================================



# 統一補完判定（共通化）
def is_unified_predictions(all_model_predictions, model_names, top_n=10):
    if not model_names or not all_model_predictions:
        return False

    first_model = model_names[0]
    first_preds = all_model_predictions.get(first_model, [])
    if not first_preds:
        return False

    base_tokens = [
        c['token']
        for c in first_preds[0]['candidates'][:top_n]
    ]

    for model in model_names:
        preds = all_model_predictions.get(model, [])
        if not preds:
            return False

        for p in preds:
            tokens = [c['token'] for c in p['candidates'][:top_n]]
            if tokens != base_tokens:
                return False

    return True

# アンサンブル統合（3関数を1つに）
def calculate_ensemble_scores(all_model_predictions,
                              model_names,
                              mode="rank",
                              alpha=0.5):
    """
    mode:
        "rank"
        "probability"
        "hybrid"
    """

    if not all_model_predictions:
        return []

    num_masks = min(len(v) for v in all_model_predictions.values())

    ensemble_results = []

    for mask_idx in range(num_masks):

        token_scores = {}

        for model in model_names:
            candidates = all_model_predictions[model][mask_idx]['candidates']

            for rank, cand in enumerate(candidates):
                token = cand['token']
                prob = cand.get('probability', 0)

                if token not in token_scores:
                    token_scores[token] = 0

                if mode == "rank":
                    token_scores[token] += 1 / (rank + 1)

                elif mode == "probability":
                    token_scores[token] += prob

                elif mode == "hybrid":
                    token_scores[token] += (
                        alpha * (1 / (rank + 1))
                        + (1 - alpha) * prob
                    )

        sorted_tokens = sorted(
            token_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        ensemble_results.append(sorted_tokens)

    return ensemble_results


# アンサンブル表示共通化
def build_ensemble_output(ensemble_results,
                          line_num,
                          process_count,
                          original_text,
                          mask_type=None,
                          group_lines=None,
                          method=None):

    lines = []

    '''
    # ヘッダー
    if method == "rank":
        lines.append("-" * 80)
        lines.append("アンサンブル補完結果（方式A: 順位ベース）")
#        lines.append("=" * 80)
        lines.append("【スコアリング方式】")
        lines.append(f"順位スコア（{RANK_SCORE_METHOD}方式）:")
        
        for i in range(1, min(MAX_RANK_TO_CONSIDER + 1, 11)):
            score = get_rank_score(i, RANK_SCORE_METHOD)
            lines.append(f"  {i}位: {score:.2f}点")
        
        lines.append(f"確率閾値: {MIN_PROBABILITY_THRESHOLD*100:.1f}%以上, 評価範囲: 1-{MAX_RANK_TO_CONSIDER}位")
        lines.append(f"参加モデル数: {len(COMPARE_MODELS)}")
#        lines.append("=" * 80)
    
    elif method == "probability":
        lines.append("-" * 80)
        lines.append("アンサンブル補完結果（方式B: 確率ベース）")
#        lines.append("=" * 80)
        lines.append("【スコアリング方式】")
        lines.append("各モデルの確率を平均")
        lines.append(f"確率閾値: {MIN_PROBABILITY_THRESHOLD*100:.1f}%以上, 評価範囲: 1-{MAX_RANK_TO_CONSIDER}位")
        lines.append(f"参加モデル数: {len(COMPARE_MODELS)}")
#        lines.append("=" * 80)
    
    elif method == "hybrid":
        lines.append("-" * 80)
        lines.append("アンサンブル補完結果（方式C: ハイブリッド）")
#        lines.append("=" * 80)
        lines.append("【スコアリング方式】")
        lines.append(f"順位スコア × 確率（{RANK_SCORE_METHOD}方式）:")
        
        for i in range(1, min(MAX_RANK_TO_CONSIDER + 1, 11)):
            score = get_rank_score(i, RANK_SCORE_METHOD)
            lines.append(f"  {i}位: {score:.2f}点")
        
        lines.append(f"確率閾値: {MIN_PROBABILITY_THRESHOLD*100:.1f}%以上, 評価範囲: 1-{MAX_RANK_TO_CONSIDER}位")
        lines.append(f"参加モデル数: {len(COMPARE_MODELS)}")
#        lines.append("=" * 80)
    '''
    
    # 処理番号とテキスト
#    lines.append("")
    lines.append("=" * 80)
    lines.append(f"{process_count}番目")
    
    if mask_type and group_lines:
        if mask_type == 'MASK':
            lines.append("通常MASKグループ:")
        else:
            lines.append(f"MASK_{mask_type}グループ:")
        
        for gline_num, gline_text in group_lines:
            lines.append(f"  行{gline_num}: {gline_text}")
    else:
        lines.append(original_text)
    
    lines.append("-" * 80)

    # 各MASKの結果
    for idx, mask_result in enumerate(ensemble_results):
        lines.append(f"[MASK]{idx+1}:")

        for rank, item in enumerate(mask_result[:ENSEMBLE_TOP_N], 1):
            if isinstance(item, dict):
                token = item['token']
                score = item['score']
                support = item.get('support', 0)
                details = item.get('details', [])
                
                # 詳細情報の整形
                detail_strs = []
                for model_name, model_rank, prob in details:
                    abbr = MODEL_ABBREVIATIONS.get(model_name, model_name)
                    detail_strs.append(f"{abbr}: {model_rank}位({prob:.4f})")
                
                detail_text = ", ".join(detail_strs)
                
                if method == "rank":
                    lines.append(f"  {rank}位: {token} ({score:.2f}点) ← {support}モデルが支持（詳細: {detail_text}）")
                elif method == "probability":
                    lines.append(f"  {rank}位: {token} ({score:.4f}) ← {support}モデルが支持（詳細: {detail_text}）")
                elif method == "hybrid":
                    lines.append(f"  {rank}位: {token} ({score:.3f}点) ← {support}モデルが支持（詳細: {detail_text}）")
            else:
                # 古い形式（タプル）
                token, score = item
                lines.append(f"  {rank}位: {token} ({score:.6f})")

        lines.append("")

    return "\n".join(lines)


# 画面表示
def display_ensemble_results(**kwargs):
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")
    
    
    output = build_ensemble_output(
        ensemble_results,
        kwargs["line_num"],
        kwargs["process_count"],
        kwargs["original_text"],
        mask_type=mask_type,
        group_lines=group_lines,
        method=method
    )
    print(output)
    
    
# 共通フォーマット関数
def format_mask_table(mask_index, line_label, model_names, predictions, top_k=10):
    lines = []
    lines.append(f"[MASK]{mask_index}: （{line_label}）")

    # ヘッダー
    header = "      " + "".join([f"{name:<12}" for name in model_names])
    lines.append(header)

    # 各順位
    for rank in range(top_k):
        row = f"{rank+1:>2}位:  "
        for model_preds in predictions:
            if rank < len(model_preds):
                token, prob = model_preds[rank]
                row += f"{token} ({prob:.4f})".ljust(12)
            else:
                row += "".ljust(12)
        lines.append(row)

    return "\n".join(lines)

# 画面表示関数
def print_mask_result(line_number, process_count, original_text,
                      model_names, predictions, mask_index=1):

    print("=" * 80)
    print(f"行 {line_number}: {process_count}番目")
    print("-" * 80)
    print("通常MASKグループ:")
    print(f"  行{line_number}: {original_text}")
    print("-" * 80)

    table = format_mask_table(mask_index,
                              f"行{line_number}",
                              model_names,
                              predictions)
    print(table)
    print("=" * 79)

# ファイル出力
def save_mask_result(f, line_number, process_count, original_text,
                     model_names, predictions, mask_index=1):

    f.write("=" * 80 + "\n")
    f.write(f"行 {line_number}: {process_count}番目\n")
    f.write("-" * 80 + "\n")
    f.write("通常MASKグループ:\n")
    f.write(f"  行{line_number}: {original_text}\n")
    f.write("-" * 80 + "\n")

    table = format_mask_table(mask_index,
                              f"行{line_number}",
                              model_names,
                              predictions)

    f.write(table + "\n")
    f.write("=" * 79 + "\n")


def get_rank_score(rank, method=RANK_SCORE_METHOD):
    """
    順位に応じたスコアを計算
    
    Args:
        rank: 順位 (1-based)
        method: 'exponential', 'logarithmic', 'linear', 'inverse'
    
    Returns:
        float: スコア
    """
    if rank <= 0 or rank > MAX_RANK_TO_CONSIDER:
        return 0.0
    
    if method == 'exponential':
        # 指数減衰: 10 * (0.75 ^ (rank-1))
        return 10.0 * (0.75 ** (rank - 1))
    
    elif method == 'logarithmic':
        # 対数減衰: 10 / log2(rank + 1)
        return 10.0 / math.log2(rank + 1)
    
    elif method == 'linear':
        # 線形減衰: 11 - rank
        return float(MAX_RANK_TO_CONSIDER + 1 - rank)
    
    elif method == 'inverse':
        # 逆数: 10 / rank
        return 10.0 / rank
    
    else:
        return 0.0

'''
Exponential（指数減衰）: 10 × 0.75^(rank-1)
順位スコア 1位10.00  2位7.50  3位5.63  4位4.22  5位3.16  6位2.37  7位1.78  8位1.33

Logarithmic（対数減衰）: 10 / log2(rank+1)
順位スコア 1位10.00  2位8.30  3位7.27  4位6.64  5位6.14  6位5.73  7位5.39  8位5.09

Linear（線形減衰）: 11 - rank
順位スコア 1位10  2位9  3位8  4位7  5位6  6位5  7位4  8位3

Inverse（逆数）: 10 / rank
順位スコア 1位10.00  2位5.00  3位3.33  4位2.50  5位2.00  6位1.67  7位1.43  8位1.25


特徴
Exponential: バランス型。上位重視だが下位も適度に評価
Logarithmic: 順位間の差が小さい。下位も比較的重視
Linear: 全順位を均等に近く評価。下位の影響が大きい
Inverse: 1位を最重視。2位以降は急激に減衰
'''

def replace_custom_masks_with_mask(text):
    """
    [MASK_XXX] を統一マスクに置換し、位置情報を保持
    
    Returns:
        tuple: (置換後テキスト, マスクタイプリスト)
        マスクタイプリスト: [(マスクインデックス, マスクタイプ), ...]
    """
    # [MASK_XXX] を検出
    pattern = r'\[MASK_([A-Z0-9_]+)\]'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return text, []
    
    result = text
    offset = 0
    mask_info = []  # [(mask_idx, mask_type), ...]
    mask_idx = 0
    
    for match in matches:
        mask_type = match.group(1)  # "A", "ABC" など
        start = match.start() + offset
        
        # [MASK_XXX] を [MASK] に置換
        result = result[:start] + '[MASK]' + result[start + len(match.group(0)):]
        offset += len('[MASK]') - len(match.group(0))
        
        mask_info.append((mask_idx, mask_type))
        mask_idx += 1
    
    return result, mask_info
    
class ChineseMaskCompletion:
    def __init__(self, model_path, top_k=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.top_k = top_k
        self.mask_token = self.tokenizer.mask_token
    
    @staticmethod
    def get_display_width(text):
        """文字列の表示幅を計算（全角=2、半角=1）"""
        width = 0
        for char in text:
            if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
                width += 2  # 全角
            else:
                width += 1  # 半角
        return width
    
    def split_sentences(self, text):
        """文章を句点で分割"""
        sentences = re.split(r'([。！？!?])', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]
    
    def find_mask_sentence(self, sentences):
        """[MASK]を含む文を見つける"""
        for i, sent in enumerate(sentences):
            if '[MASK]' in sent:
                return i
        return -1
    
    @staticmethod
    def is_valid_chinese_token(token: str) -> bool:
        if not token:
            return False

        # UNK系・特殊トークン
        if token in {"<UNK>", "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}:
            return False
    
        # ★追加: [unused1]などのBERT特殊トークンを除外
        if token.startswith("[") and token.endswith("]"):
            return False
    
        # サブワードトークン（##接頭辞）を除外
        if token.startswith("##"):
            return False

        # 1文字でないものは除外
        if len(token) != 1:
            return False

        ch = token[0]

        # 絵文字・記号除外
        cat = unicodedata.category(ch)
        if cat.startswith("S"):  # Symbol
            return False

        # CJK統合漢字のみ許可
        if not (
            "\u4e00" <= ch <= "\u9fff" or      # CJK Unified Ideographs
            "\u3400" <= ch <= "\u4dbf" or      # CJK Extension A
            "\u20000" <= ch <= "\u2a6df" or    # CJK Extension B
            "\u2a700" <= ch <= "\u2b73f"       # CJK Extension C
        ):
            return False

        return True
    

    def evaluate_mlm_score_fast(self, text, skip_positions=None):
        """文全体のMLMスコアを計算"""
        if skip_positions is None:
            skip_positions = set()
    
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].clone()
    
        # 特殊トークンID
        special_ids = set([
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        ])
    
        # 評価対象の位置を収集
        eval_positions = []
        for i in range(input_ids.size(1)):
            token_id = input_ids[0, i].item()
            if i in skip_positions or token_id in special_ids:
                continue
            eval_positions.append(i)
    
        if not eval_positions:
            return 0.0
    
        # バッチマスク作成
        batch_size = len(eval_positions)
        batch_input_ids = input_ids.repeat(batch_size, 1)
    
        for idx, pos in enumerate(eval_positions):
            batch_input_ids[idx, pos] = self.tokenizer.mask_token_id
    
        # バッチ推論
        with torch.no_grad():
            outputs = self.model(batch_input_ids)
            logits = outputs.logits
    
        # 各位置のスコア計算
        total_score = 0.0
        for idx, pos in enumerate(eval_positions):
            original_token = input_ids[0, pos].item()
            token_logits = logits[idx, pos]
            prob = torch.softmax(token_logits, dim=0)[original_token].item()
            total_score += math.log(prob + 1e-12)
    
        return total_score
    
    def get_context_window(self, text, use_full_context=True):
        """文脈ウィンドウを取得（512トークン制約考慮）"""
        if not use_full_context:
            sentences = self.split_sentences(text)
            mask_idx = self.find_mask_sentence(sentences)
            if mask_idx >= 0:
                return sentences[mask_idx]
            return text
        
        # 512トークン制約チェック
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= MAX_TOKENS:
            return text
        
        # [MASK]位置を中心にトークンを抽出
        sentences = self.split_sentences(text)
        mask_idx = self.find_mask_sentence(sentences)
        
        if mask_idx < 0:
            # [MASK]がない場合は先頭から切り詰め
            decoded = self.tokenizer.decode(tokens[:MAX_TOKENS], skip_special_tokens=True)
            return decoded
        
        # [MASK]を含む文を中心に前後を追加
        context = [sentences[mask_idx]]
        left, right = mask_idx - 1, mask_idx + 1
        
        while left >= 0 or right < len(sentences):
            current = ''.join(context)
            current_tokens = len(self.tokenizer.encode(current, add_special_tokens=True))
            
            if current_tokens >= MAX_TOKENS:
                break
            
            if left >= 0:
                test = sentences[left] + current
                if len(self.tokenizer.encode(test, add_special_tokens=True)) <= MAX_TOKENS:
                    context.insert(0, sentences[left])
                    left -= 1
                else:
                    left = -1
            
            if right < len(sentences):
                test = current + sentences[right]
                if len(self.tokenizer.encode(test, add_special_tokens=True)) <= MAX_TOKENS:
                    context.append(sentences[right])
                    right += 1
                else:
                    right = len(sentences)
        
        return ''.join(context)
    
    def predict_masks(self, text, use_full_context=True):
        """複数の[MASK]を予測"""
        context_text = self.get_context_window(text, use_full_context)

        # トークン化
        inputs = self.tokenizer(context_text, return_tensors="pt").to(self.device)

        # マスクトークンのインデックスを取得
        mask_token_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_token_indices) == 0:
            return []

        # 予測
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        results = []
        for mask_idx, token_idx in enumerate(mask_token_indices):
            mask_logits = predictions[0, token_idx]
            probs = torch.softmax(mask_logits, dim=0)
        
            # ★修正: 語彙サイズチェック
            vocab_size = len(probs)
            OVERGEN_K = min(self.top_k * 10, vocab_size)  # ★追加
        
            top_probs, top_indices = torch.topk(probs, OVERGEN_K)

            candidates = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.convert_ids_to_tokens(int(idx))
        
                if not self.is_valid_chinese_token(token):
                    continue
            
                candidates.append({
                    'token': token,
                    'probability': prob.item()
                })
        
                if len(candidates) >= self.top_k:
                    break
    
            results.append({
                'position': mask_idx,
                'candidates': candidates
            })

        return results

def group_consecutive_lines(lines, mask_type='MASK'):
    groups = []
    current_group = []

    for i, line in enumerate(lines, 1):
        has_mask = '[MASK]' in line if mask_type=='MASK' else f'[MASK_{mask_type}]' in line

        if has_mask:
            # 通常MASKは1行ずつ
            if mask_type == 'MASK':
                if current_group:
                    combined_text = "\n".join(l for _, l in current_group)
                    groups.append((current_group, False, combined_text))
                    current_group = []
                current_group.append((i, line))
                combined_text = line
                groups.append((current_group, False, combined_text))
                current_group = []
            else:
                current_group.append((i, line))
        else:
            if current_group:
                combined_text = "\n".join(l for _, l in current_group)
                is_combined = len(current_group) >= 2
                groups.append((current_group, is_combined, combined_text))
                current_group = []

    if current_group:
        combined_text = "\n".join(l for _, l in current_group)
        is_combined = len(current_group) >= 2
        groups.append((current_group, is_combined, combined_text))

    return groups


def calculate_ensemble_scores_rank(all_model_predictions, model_names):
    """方式A: 順位ベーススコアリング（動的スコア計算）"""
    if not model_names or not all_model_predictions:
        return []
    
    # ★修正: max()ではなく最初のモデルの予測数を使用
    num_masks = len(next(iter(all_model_predictions.values()), []))
    ensemble_results = []
    
    
    
    # ===== 統一補完判定 =====
    is_unified = False

    if len(all_model_predictions) > 0:
        first_model = model_names[0]
        predictions = all_model_predictions.get(first_model, [])

        if len(predictions) >= 1:
            first_tokens = [c['token'] for c in predictions[0]['candidates'][:10]]

            all_same = True
            for model_name in model_names:
                model_preds = all_model_predictions.get(model_name, [])
                for pred in model_preds:
                    tokens = [c['token'] for c in pred['candidates'][:10]]
                    if tokens != first_tokens:
                        all_same = False
                        break
                if not all_same:
                    break

            is_unified = all_same
        
    
#    for mask_idx in range(num_masks):
#    display_count = 1 if is_unified else num_masks

    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions.get(model_name, [])
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                if prob < MIN_PROBABILITY_THRESHOLD:
                    continue
                
                score = get_rank_score(rank, RANK_SCORE_METHOD)
                
                if token not in token_scores:
                    token_scores[token] = {"score": 0, "models": []}
                
                token_scores[token]["score"] += score
                token_scores[token]["models"].append((model_name, rank, prob))
        
        # ★追加: 支持数フィルタ
        filtered_tokens = {
            token: data 
            for token, data in token_scores.items() 
            if len(data["models"]) >= MIN_SUPPORT
        }
        
        sorted_tokens = sorted(
            filtered_tokens.items(), 
            key=lambda x: (-x[1]["score"], x[0])
        )
        
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({
                "token": token,
                "score": data["score"],
                "support": len(data["models"]),
                "details": data["models"]
            })
        
        ensemble_results.append(mask_result)
    
    return ensemble_results


def calculate_ensemble_scores_probability(all_model_predictions, model_names):
    """方式B: 確率ベーススコアリング"""
    if not model_names or not all_model_predictions:
        return []
    
    # ★修正: max()ではなく最初のモデルの予測数を使用
    num_masks = len(next(iter(all_model_predictions.values()), []))
    ensemble_results = []
    
    
    
        # ===== 統一補完判定 =====
    is_unified = False

    if len(all_model_predictions) > 0:
        first_model = model_names[0]
        predictions = all_model_predictions.get(first_model, [])

        if len(predictions) >= 1:
            first_tokens = [c['token'] for c in predictions[0]['candidates'][:10]]

            all_same = True
            for model_name in model_names:
                model_preds = all_model_predictions.get(model_name, [])
                for pred in model_preds:
                    tokens = [c['token'] for c in pred['candidates'][:10]]
                    if tokens != first_tokens:
                        all_same = False
                        break
                if not all_same:
                    break

            is_unified = all_same
        
    
#    for mask_idx in range(num_masks):
#    display_count = 1 if is_unified else num_masks

    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions.get(model_name, [])
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                if prob < MIN_PROBABILITY_THRESHOLD:
                    continue
                
                if token not in token_scores:
                    token_scores[token] = {"probs": [], "models": []}
                
                token_scores[token]["probs"].append(prob)
                token_scores[token]["models"].append((model_name, rank, prob))
        
        # 平均確率を計算
        for token in token_scores:
            avg_prob = sum(token_scores[token]["probs"]) / len(token_scores[token]["probs"])
            token_scores[token]["score"] = avg_prob
        
        # 支持数フィルタ適用
        filtered_tokens = {
            token: data 
            for token, data in token_scores.items() 
            if len(data["models"]) >= MIN_SUPPORT
        }
        
        sorted_tokens = sorted(
            filtered_tokens.items(),
            key=lambda x: (-x[1]["score"], x[0])
        )
        
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({
                "token": token,
                "score": data["score"],
                "support": len(data["models"]),
                "details": data["models"]
            })
        
        ensemble_results.append(mask_result)
    
    return ensemble_results


def calculate_ensemble_scores_hybrid(all_model_predictions, model_names):
    """方式C: ハイブリッドスコアリング（動的スコア計算）"""
    if not model_names or not all_model_predictions:
        return []
    
    # ★修正: max()ではなく最初のモデルの予測数を使用
    num_masks = len(next(iter(all_model_predictions.values()), []))
    ensemble_results = []
    
    
        # ===== 統一補完判定 =====
    is_unified = False

    if len(all_model_predictions) > 0:
        first_model = model_names[0]
        predictions = all_model_predictions.get(first_model, [])

        if len(predictions) >= 1:
            first_tokens = [c['token'] for c in predictions[0]['candidates'][:10]]

            all_same = True
            for model_name in model_names:
                model_preds = all_model_predictions.get(model_name, [])
                for pred in model_preds:
                    tokens = [c['token'] for c in pred['candidates'][:10]]
                    if tokens != first_tokens:
                        all_same = False
                        break
                if not all_same:
                    break

            is_unified = all_same
        
    
#    for mask_idx in range(num_masks):
#    display_count = 1 if is_unified else num_masks

    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions.get(model_name, [])
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                if prob < MIN_PROBABILITY_THRESHOLD:
                    continue
                
                rank_score = get_rank_score(rank, RANK_SCORE_METHOD)
                
                if token not in token_scores:
                    token_scores[token] = {"score": 0, "models": []}
                
                hybrid_score = rank_score * prob
                token_scores[token]["score"] += hybrid_score
                token_scores[token]["models"].append((model_name, rank, prob))
        
        # 支持数フィルタ適用
        filtered_tokens = {
            token: data 
            for token, data in token_scores.items() 
            if len(data["models"]) >= MIN_SUPPORT
        }
        
        sorted_tokens = sorted(
            filtered_tokens.items(),
            key=lambda x: (-x[1]["score"], x[0])
        )
        
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({
                "token": token,
                "score": data["score"],
                "support": len(data["models"]),
                "details": data["models"]
            })
        
        ensemble_results.append(mask_result)
    
    return ensemble_results



# ファイル保存（画面と完全一致）
def save_ensemble_results(output_file, **kwargs):
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")

    output = build_ensemble_output(
        ensemble_results,
        kwargs["line_num"],
        kwargs["process_count"],
        kwargs["original_text"],
        mask_type=mask_type,
        group_lines=group_lines,
        method=method
    )

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(output + "\n")


def display_comparison_results(all_model_predictions, model_names, line_text,
                               line_num=None, process_count=None,
                               show_prob=True, decimals=4,
                               mask_type=None, group_lines=None,
                               f=None):

    lines = []
    
    
    lines.append("=" * 80)
    lines.append(f"{process_count}番目")
    lines.append("-" * 80)

    if mask_type and group_lines:
        if mask_type == 'MASK':
            lines.append("通常MASKグループ:")
        else:
            lines.append(f"MASK_{mask_type}グループ:")

        for gline_num, gline_text in group_lines:
            lines.append(f"  行{gline_num}: {gline_text}")

        lines.append("-" * 80)
    else:
        lines.append(line_text)
        lines.append("-" * 80)

    header = "      " + "".join([f"{name:<12}" for name in model_names])
    lines.append(header)

    # ---- 文字列データを除外（重要）----
    filtered_predictions = [
        p for p in all_model_predictions
        if isinstance(p, list) and len(p) > 0
    ]

    if not filtered_predictions:
        lines.append("予測結果なし")
        text = "\n".join(lines)
        print(text)
        if f is not None:
            f.write(text + "\n")
        return

    max_rank = max(len(p) for p in filtered_predictions)

    for rank in range(max_rank):
        row = f"{rank+1:>2}位:  "

        for model_preds in filtered_predictions:

            if rank < len(model_preds):

                item = model_preds[rank]

                # ---- 安全処理ここ ----
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    token, prob = item
                    if show_prob:
                        row += f"{token} ({prob:.{decimals}f})".ljust(12)
                    else:
                        row += f"{token}".ljust(12)

                else:
                    # tokenのみのケース
                    row += f"{item}".ljust(12)

            else:
                row += "".ljust(12)

        lines.append(row)

    text = "\n".join(lines)

    if f is None:
        print("\n" + text)
#        print("=" * 80)  # 画面表示のみ最後に=を追加
    else:
        f.write(text + "\n\n")
#        f.write("=" * 80 + "\n")  # ファイル出力に=を追加


            
def save_comparison_to_txt(all_results, output_file, show_prob=True, decimals=6):
    """比較結果をTSV形式で保存"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for line_num, process_count, original_text, all_model_predictions, model_names in all_results:
            f.write("=" * 80 + "\n")
            f.write(f"{process_count}番目\n")
            f.write("-" * 80 + "\n")
            f.write(f"{original_text}\n")
            f.write("-" * 80 + "\n")

            if not model_names or not all_model_predictions:
                f.write("候補なし\n\n")
                continue

            # ★修正: 最初のモデルの予測数を使用（全モデル同じはず）
            num_masks = len(next(iter(all_model_predictions.values()), []))
            
            
            # ===== 統一補完判定 =====
            is_unified = False

            if len(all_model_predictions) > 0:
                first_model = model_names[0]
                predictions = all_model_predictions.get(first_model, [])

                if len(predictions) >= 1:
                    first_tokens = [c['token'] for c in predictions[0]['candidates'][:10]]

                    all_same = True
                    for model_name in model_names:
                        model_preds = all_model_predictions.get(model_name, [])
                        for pred in model_preds:
                            tokens = [c['token'] for c in pred['candidates'][:10]]
                            if tokens != first_tokens:
                                all_same = False
                                break
                        if not all_same:
                            break

                    is_unified = all_same


#            for mask_idx in range(num_masks):
            display_count = 1 if is_unified else num_masks
            for mask_idx in range(display_count):
                if mask_idx > 0:
                    f.write("\n")

                f.write(f"[MASK]{mask_idx + 1}:\n")

                # ===== TSV ヘッダー =====
                header = ["順位"]
                for model_name in model_names:
                    header.append(MODEL_ABBREVIATIONS.get(model_name, model_name))
                f.write("\t".join(header) + "\n")

                # ===== 各モデルの最大セル幅を事前計算 =====
                max_widths = []
                for model_name in model_names:
                    max_width = ChineseMaskCompletion.get_display_width(MODEL_ABBREVIATIONS.get(model_name, model_name))
                    preds = all_model_predictions.get(model_name, [])
                    if mask_idx < len(preds):
                        for rank in range(TOP_K):
                            if rank < len(preds[mask_idx]["candidates"]):
                                cand = preds[mask_idx]["candidates"][rank]
                                if show_prob:
                                    cell = f"{cand['token']} ({cand['probability']:.{decimals}f})"
                                else:
                                    cell = cand["token"]
                                width = ChineseMaskCompletion.get_display_width(cell)
                                max_width = max(max_width, width)
                    max_widths.append(max_width)
                
                # ===== TSV 本体（空白で列位置を揃える）=====
                for rank in range(TOP_K):
                    row = [f"{rank+1:2d}位:"]
                    for i, model_name in enumerate(model_names):
                        preds = all_model_predictions.get(model_name, [])
                        if mask_idx < len(preds) and rank < len(preds[mask_idx]["candidates"]):
                            cand = preds[mask_idx]["candidates"][rank]
                            if show_prob:
                                cell = f"{cand['token']} ({cand['probability']:.{decimals}f})"
                            else:
                                cell = cand["token"]
                        else:
                            cell = ""
                        
                        # 列幅に合わせてパディング
                        padding = max_widths[i] - ChineseMaskCompletion.get_display_width(cell)
                        row.append(cell + " " * padding)
                    
                    f.write("\t".join(row) + "\n")
               
            f.write("\n")

def main():
    # コマンドライン引数チェック
    if len(sys.argv) < 2:
        print("使い方: python script.py <入力テキストファイル>")
        print("例: python script.py input.txt")
        return
    
    input_file = sys.argv[1]
    
    # 出力ファイル名を生成
    input_path = Path(input_file)
    output_file = input_path.stem + "_out" + input_path.suffix
    
    # ★既存の出力ファイルを削除（初期化）
    if Path(output_file).exists():
        Path(output_file).unlink()
    
    # ★アンサンブル結果ファイルも削除
    for method in ENSEMBLE_METHODS:
        output_file_method = output_file.replace('.txt', f'_{method}.txt')
        if Path(output_file_method).exists():
            Path(output_file_method).unlink()
    
    # モデルをロード
    models = {}
    for model_name in COMPARE_MODELS:
        model_path = f"models/{model_name}"
        if not Path(model_path).exists():
            print(f"警告: モデルが見つかりません: {model_path} (スキップします)")
            continue
        models[model_name] = ChineseMaskCompletion(model_path, top_k=TOP_K)
    
    if not models:
        print("エラー: 使用可能なモデルがありません")
        return
    
    # ★追加: モデル順序を固定化（以降変更禁止）
    model_names = list(models.keys())
    
    # 入力ファイル読み込み
    if not Path(input_file).exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # ===== ステップ1: 全行を読み込み、マスクタイプごとに分類 =====
    mask_groups = {}
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        original_line = line
        
        # カスタムマスクのパターン
        custom_mask_pattern = r'\[MASK_([A-Z0-9_]+)\]'
        
        # カスタムマスク（[MASK_XXX]）を検出
        custom_matches = re.findall(custom_mask_pattern, line)
        
        # 通常の[MASK]を検出（カスタムマスクを一時的に除外して確認）
        temp_line = re.sub(custom_mask_pattern, '##TEMP##', line)
        has_normal_mask = '[MASK]' in temp_line
        
        # カスタムマスクを処理
        if custom_matches:
            unique_custom = list(dict.fromkeys(custom_matches))
            
            for mask_type in unique_custom:
                # このタイプのマスクだけを[MASK]に置換（他のマスクは保持）
                processed_line = re.sub(rf'\[MASK_{mask_type}\]', '[MASK]', original_line)
                
                if mask_type not in mask_groups:
                    mask_groups[mask_type] = []
                mask_groups[mask_type].append({
                    'line_num': line_num,
                    'original': original_line,
                    'processed': processed_line
                })
        
        # 通常の[MASK]を処理
        if has_normal_mask:
            # カスタムマスクをすべて削除（空文字に置換）
            processed_line = re.sub(custom_mask_pattern, '', original_line)
            
            if 'MASK' not in mask_groups:
                mask_groups['MASK'] = []
            mask_groups['MASK'].append({
                'line_num': line_num,
                'original': original_line,
                'processed': processed_line
            })
    
    # ===== ステップ2: グループごとに処理 =====
    all_results = []
    process_count = 0
    
    # 文ペア評価用モデル（信頼できるモデルのみ）
    PAIR_EVAL_MODELS = {
        'hfl_chinese-macbert-large',
        'hfl_chinese-roberta-wwm-ext',
        'bert-base-chinese'
    }
    
    # 候補プール設定
    CANDIDATE_POOL_TOP_K = 30  # 各文から取得する候補数
    
    for mask_type, entries in mask_groups.items():
        # ============================================================
        # 通常MASKの処理ブロック
        # ============================================================
        if mask_type == 'MASK':
            # 1行ずつ処理
            for entry in entries:
                process_count += 1
                
                # 1行のみのグループとして処理
                single_entry = [entry]
                line_nums = [entry['line_num']]
                group_lines = [(entry['line_num'], entry['original'])]
                
                # グループ情報を表示
                print(f"\n処理中: 通常MASK（単一行）: {process_count}番目")
                print(f"対象行: 行{entry['line_num']}")
                
                # 表示用テキスト
                combined_original = entry['original']
                
                # 通常処理
                combined_processed = entry['processed']
                
                # 各モデルで予測
                all_model_predictions = {}
                for model_name, model in models.items():
                    predictions = model.predict_masks(combined_processed, use_full_context=USE_FULL_LINE_CONTEXT)
                    all_model_predictions[model_name] = predictions
                
                # データ形式を変換（辞書 → リスト）
                predictions_for_display = []
                for model_name in model_names:
                    if model_name in all_model_predictions:
                        preds = all_model_predictions[model_name]
                        if len(preds) > 0:
                            candidates = preds[0].get('candidates', [])
                            predictions_for_display.append([
                                (c['token'], c['probability']) 
                                for c in candidates[:TOP_K]
                            ])
                
                # 各モデルの予測を横並び表示（画面）
                model_abbr = [MODEL_ABBREVIATIONS.get(name, name) for name in model_names]
                display_comparison_results(
                    predictions_for_display,
                    model_abbr,
                    combined_original,
                    line_num=entry['line_num'],
                    process_count=process_count,
                    show_prob=True,
                    decimals=4,
                    mask_type=mask_type,
                    group_lines=group_lines,
                    f=None
                )
                
                # ★ファイル出力を追加
                with open(output_file, 'a', encoding='utf-8') as f:
                    display_comparison_results(
                        predictions_for_display,
                        model_abbr,
                        combined_original,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        show_prob=True,
                        decimals=PROB_DECIMALS_FILE,
                        mask_type=mask_type,
                        group_lines=group_lines,
                        f=f
                    )
                
                # アンサンブル計算（3方式）
                for method in ENSEMBLE_METHODS:
#                    print(f"\n--- アンサンブル方式: {method.upper()} ---")
                    
                    if method == "rank":
                        ensemble_results = calculate_ensemble_scores_rank(all_model_predictions, list(models.keys()))
                    elif method == "probability":
                        ensemble_results = calculate_ensemble_scores_probability(all_model_predictions, list(models.keys()))
                    elif method == "hybrid":
                        ensemble_results = calculate_ensemble_scores_hybrid(all_model_predictions, list(models.keys()))
                    
                    # 表示
                    display_ensemble_results(
                        ensemble_results=ensemble_results,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        original_text=entry['original'],
                        mask_type=mask_type,
                        group_lines=group_lines,
                        method=method
                    )
                    
                    # 保存
                    output_file_method = output_file.replace('.txt', f'_{method}.txt')
                    save_ensemble_results(
                        output_file=output_file_method,
                        ensemble_results=ensemble_results,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        original_text=entry['original'],
                        mask_type=mask_type,
                        group_lines=group_lines,
                        method=method
                    )
                
                # ★forループの外：all_resultsへの追加（1回のみ）
                all_results.append((entry['line_num'], process_count, combined_original, all_model_predictions, list(models.keys())))
        
        # ============================================================
        # カスタムマスクの処理ブロック
        # ============================================================
        else:
            process_count += 1

            # 行番号リストとグループ情報を作成
            line_nums = [entry['line_num'] for entry in entries]
            group_lines = [(entry['line_num'], entry['original']) for entry in entries]

            # グループ情報を表示
            if len(entries) >= 2:
                print(f"\n処理中: MASK_{mask_type}グループ（{len(entries)}行連結）: {process_count}番目")
            else:
                print(f"\n処理中: MASK_{mask_type}（単一行）: {process_count}番目")
            print(f"対象行: " + ", ".join(f"行{num}" for num in line_nums))

            # 表示用テキスト
            combined_original = '\n'.join(entry['original'] for entry in entries)

            # ===== 各行のマスク数をカウント =====
            mask_counts = [entry['processed'].count('[MASK]') for entry in entries]
            total_masks = sum(mask_counts)

            # ===== カスタムマスクグループ（2行以上 OR 1行2マスク以上）の場合は統一候補生成 =====
            if len(entries) >= 2 or total_masks >= 2:
                
                # 文ペア評価用モデル（信頼できるモデルのみ）
                PAIR_EVAL_MODELS = {
                    'hfl_chinese-macbert-large',
                    'hfl_chinese-roberta-wwm-ext',
                    'bert-base-chinese'
                }
                
                # 候補プール設定
                CANDIDATE_POOL_TOP_K = 50  # ★拡大: 30 → 50
                
                # ===== ステップ1: 各MASK位置での候補プールを取得 =====
                all_mask_candidates = {}  # {token: {prob_sum, count, details}}
                
                for entry in entries:
                    # カスタムマスクを削除（通常MASKのみ残す）
                    clean_processed = re.sub(r'\[MASK_[A-Z0-9_]+\]', '', entry['processed'])
                    
                    # 各モデルで予測
                    for model_name, model in models.items():
                        if model_name not in PAIR_EVAL_MODELS:
                            continue
                        
                        preds = model.predict_masks(clean_processed, use_full_context=USE_FULL_LINE_CONTEXT)
                        if not preds:
                            continue
                        
                        for cand in preds[0]['candidates'][:CANDIDATE_POOL_TOP_K]:
                            token = cand['token']
                            prob = cand['probability']
                            
                            if token not in all_mask_candidates:
                                all_mask_candidates[token] = {
                                    'prob_sum': 0.0,
                                    'count': 0,
                                    'details': []
                                }
                            
                            all_mask_candidates[token]['prob_sum'] += prob
                            all_mask_candidates[token]['count'] += 1
                            all_mask_candidates[token]['details'].append((model_name, prob))
                
                # ===== ステップ2: サポート数フィルタリング =====
                min_support = max(2, len(PAIR_EVAL_MODELS))  # 最低2モデル
                candidate_pool_filtered = {
                    token: data 
                    for token, data in all_mask_candidates.items() 
                    if data['count'] >= min_support and re.fullmatch(r'[\u4e00-\u9fff]+', token)
                }
                
                if not candidate_pool_filtered:
                    # フィルタ後に候補がない場合
                    candidate_pool_filtered = {
                        token: data 
                        for token, data in all_mask_candidates.items() 
                        if re.fullmatch(r'[\u4e00-\u9fff]+', token)
                    }
                
                # ===== ステップ3: PLL評価 =====
                eval_model_name = 'hfl_chinese-macbert-large'
                eval_model = models[eval_model_name]
                
                rescored_candidates = []
                
                for token, data in candidate_pool_filtered.items():
                    # 元の平均確率
                    avg_prob = data['prob_sum'] / data['count']
                    
                    # 各MASK位置にトークンを当てはめた文を作成
                    filled_texts = []
                    for entry in entries:
                        clean_processed = re.sub(r'\[MASK_[A-Z0-9_]+\]', '', entry['processed'])
                        filled = clean_processed.replace('[MASK]', token, 1)
                        filled_texts.append(filled)
                    
                    # 全文を結合
                    combined_filled = '\n'.join(filled_texts)
                    
                    # MLMスコア計算
                    pll_score = eval_model.evaluate_mlm_score_fast(combined_filled, skip_positions=set())
                    
                    # ★修正: 混合スコア（PLL 50% + 元確率 50%）
                    # PLLスコアを[-50, 0]の範囲と想定し、[0, 1]に正規化
                    normalized_pll = (pll_score + 50.0) / 50.0
                    normalized_pll = max(min(normalized_pll, 1.0), 0.0)  # [0, 1]にクリップ
                    
                    # 混合スコア（PLL重視）
                    hybrid_score = 0.6 * normalized_pll + 0.4 * avg_prob  # ★修正: PLL 60%
                    
                    rescored_candidates.append({
                        'token': token,
                        'probability': hybrid_score,
                        'pll_score': pll_score,
                        'avg_prob': avg_prob
                    })
                
                # ===== ステップ4: ソートと正規化 =====
                rescored_candidates.sort(key=lambda x: -x['probability'])
                
                # ★修正: 温度パラメータを使った穏やかな正規化
                if rescored_candidates:
                    temperature = 0.2  # ★修正: 0.5 → 0.2（さらに鋭くする）
                    
                    max_score = max(x['probability'] for x in rescored_candidates)
                    exp_scores = [math.exp((x['probability'] - max_score) / temperature) for x in rescored_candidates]
                    total = sum(exp_scores)
                    
                    for i, item in enumerate(rescored_candidates):
                        item['probability'] = exp_scores[i] / total if total > 0 else 0.0
                
                unified_candidates = rescored_candidates[:TOP_K]
                
                # ===== ステップ5: 全モデルに統一候補を適用 =====
                all_model_predictions = {}
                if not unified_candidates:
                    print("警告: 統一候補の生成に失敗。フォールバック処理")
                    for model_name, model in models.items():
                        predictions = model.predict_masks(entries[0]['processed'], use_full_context=USE_FULL_LINE_CONTEXT)
                        all_model_predictions[model_name] = predictions * total_masks if predictions else []
                else:
                    for model_name in models.keys():
                        all_model_predictions[model_name] = [
                            {'position': i, 'candidates': unified_candidates}
                            for i in range(total_masks)
                        ]
                
                # データ形式を変換（辞書 → リスト）
                predictions_for_display = []
                for model_name in model_names:
                    if model_name in all_model_predictions:
                        preds = all_model_predictions[model_name]
                        if len(preds) > 0:
                            candidates = preds[0].get('candidates', [])
                            predictions_for_display.append([
                                (c['token'], c['probability']) 
                                for c in candidates[:TOP_K]
                            ])

            else:
                # 通常処理
                combined_processed = '\n'.join(entry['processed'] for entry in entries)
                all_model_predictions = {}
                for model_name, model in models.items():
                    all_model_predictions[model_name] = model.predict_masks(combined_processed, use_full_context=USE_FULL_LINE_CONTEXT)
                
                # データ形式を変換
                predictions_for_display = []
                for model_name in model_names:
                    if model_name in all_model_predictions:
                        preds = all_model_predictions[model_name]
                        if len(preds) > 0:
                            candidates = preds[0].get('candidates', [])
                            predictions_for_display.append([
                                (c['token'], c['probability']) 
                                for c in candidates[:TOP_K]
                            ])
            
            # ★★★ if/elseの外：各モデルの予測を横並び表示（画面）★★★
            model_abbr = [MODEL_ABBREVIATIONS.get(name, name) for name in model_names]
            display_comparison_results(
                predictions_for_display,
                model_abbr,
                combined_original,
                line_num=line_nums[0],
                process_count=process_count,
                show_prob=True,
                decimals=4,
                mask_type=mask_type,
                group_lines=group_lines,
                f=None
            )
            
            # ★ファイル出力を追加
            with open(output_file, 'a', encoding='utf-8') as f:
                display_comparison_results(
                    predictions_for_display,
                    model_abbr,
                    combined_original,
                    line_num=line_nums[0],
                    process_count=process_count,
                    show_prob=True,
                    decimals=PROB_DECIMALS_FILE,
                    mask_type=mask_type,
                    group_lines=group_lines,
                    f=f
                )

            # ===== アンサンブル計算・表示・保存（3方式）=====
            for method in ENSEMBLE_METHODS:
#                print(f"\n--- アンサンブル方式: {method.upper()} ---")
                
                if method == "rank":
                    ensemble_results = calculate_ensemble_scores_rank(all_model_predictions, list(models.keys()))
                elif method == "probability":
                    ensemble_results = calculate_ensemble_scores_probability(all_model_predictions, list(models.keys()))
                elif method == "hybrid":
                    ensemble_results = calculate_ensemble_scores_hybrid(all_model_predictions, list(models.keys()))
                
                # 表示
                display_ensemble_results(
                    ensemble_results=ensemble_results,
                    line_num=line_nums[0],
                    process_count=process_count,
                    original_text=combined_original,
                    mask_type=mask_type,
                    group_lines=group_lines,
                    method=method
                )
                
                # 保存（方式名を追加）
                output_file_method = output_file.replace('.txt', f'_{method}.txt')
                save_ensemble_results(
                    output_file=output_file_method,
                    ensemble_results=ensemble_results,
                    line_num=line_nums[0],
                    process_count=process_count,
                    original_text=combined_original,
                    mask_type=mask_type,
                    group_lines=group_lines,
                    method=method
                )
            
            # ★forループの外：all_resultsへの追加（1回のみ）
            all_results.append((line_nums[0], process_count, combined_original, all_model_predictions, list(models.keys())))
            
if __name__ == "__main__":
    main()