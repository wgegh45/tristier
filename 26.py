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
    # KLUE系（CC BY-SA 4.0）
    "klue/roberta-base",
    "klue/roberta-large",
    "klue/roberta-small", 
    "klue/bert-base",

    # Beomi系
    "beomi/kcbert-base",
### "beomi/KcELECTRA-base",  # 除外（語彙の99.6%が非ハングル）
    "beomi/KcELECTRA-base-v2022",
    
    # ELECTRA系
    "monologg/koelectra-base-v3-discriminator",
    "tunib/electra-ko-base", 
    
    # その他
    "lassl/bert-ko-base",
    "jinmang2/kpfbert", 
    "monologg/distilkobert",
### "monologg/kobert"
### "kykim/bert-kor-base",
]

# モデル名の略称
MODEL_ABBREVIATIONS = {
    # KLUE系
    "klue/roberta-base": "KR-b",  # KLUE-RoBERTa（推奨）　CC-BY-SA-4.0 licenseのため再学習不可　準主軸（安定）1.2
    "klue/roberta-large": "KR-l", # KLUE-RoBERTa大型版　　CC-BY-SA-4.0 licenseのため再学習不可　最上位（主軸）1.5
    "klue/roberta-small": "KR-s", # KLUE-RoBERTa軽量版　　　　　　　　　　　　　　　　　　　　　
    "klue/bert-base": "KB-b",     # KLUE-BERT　　　　　　 CC-BY-SA-4.0 licenseのため再学習不可　

    # Beomi系
    "beomi/kcbert-base": "KC-b",  # KcBERT（SNS、コメント特化）　　　　　　　　　　　　　　補助
### "beomi/KcELECTRA-base": "KcE-b",         # KcELECTRA
    "beomi/KcELECTRA-base-v2022": "KcE-22",  # KcELECTRA v2022
    
    # ELECTRA系
    "monologg/koelectra-base-v3-discriminator": "KE-v3",  # KoELECTRA v3 最高性能 精度低い　補助 1.3
    "tunib/electra-ko-base": "TE-b",                      # TUNiB ELECTRA 100GB学習
    
    # その他
    "lassl/bert-ko-base": "LS-b",        # LASSL BERT
    "jinmang2/kpfbert": "KPF-b",         # KPF BERT　ニュース特化
    "monologg/distilkobert": "DK-b",     # DistilKoBERT　軽量28.4M
### "kykim/bert-kor-base": "KK-b",# KoBERT  MOUのため使用不可    
### "monologg/kobert": "KoB"      # KoBERT（別実装） トークナイザーに互換性問題があるため使用不可

}

# 候補数設定
TOP_K = 100              # 内部処理用（字母フィルタリング用に多めに保持） 変更前50
TOP_K_DISPLAY = 10      # 画面・ファイル表示用

# 文脈考慮モード（True: 行全体を考慮, False: 1文のみ考慮）
USE_FULL_LINE_CONTEXT = True

# 表示モード（"vertical": 縦並び, "horizontal": 横並び）
DISPLAY_MODE = "horizontal"

# 確率表示（True: 表示, False: 非表示）
SHOW_PROBABILITY = True

# 確率の小数点以下桁数
PROB_DECIMALS_DISPLAY = 3  # 画面表示用
PROB_DECIMALS_FILE = 2     # ファイル出力用

# モデルの最大トークン数
MAX_TOKENS = 512

# 字母補完モード（True: 有効, False: 無効）
JAMO_MODE = True

# 温度パラメータ（1.0=デフォルト、>1.0で確率分布を平坦化、多様な候補）
TEMPERATURE = 1.2

# 文ペア評価設定
CANDIDATE_POOL_TOP_K = 30  # 各文から取得する候補数（候補プール用）

# ==================== アンサンブルスコアリング設定 ====================
MIN_PROBABILITY_THRESHOLD = 0.015  # 1.5%未満は除外
MAX_RANK_TO_CONSIDER = 10  # 評価する最大順位

# 使用するスコア計算方式
RANK_SCORE_METHOD = 'exponential'  # 'exponential', 'logarithmic', 'linear', 'inverse'

# アンサンブル結果の表示候補数
ENSEMBLE_TOP_N = 3

# 使用する方式の選択
ENSEMBLE_METHODS = ["rank", "probability", "hybrid"]
# =====================================================================

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


# 韓国語字母の定義
CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSUNG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

def get_chosung(char):
    """ハングル文字から初声を抽出"""
    if not ('가' <= char <= '힣'):
        return None
    code = ord(char) - ord('가')
    chosung_index = code // (21 * 28)
    return CHOSUNG[chosung_index]

def get_jungsung(char):
    """ハングル文字から中声を抽出"""
    if not ('가' <= char <= '힣'):
        return None
    code = ord(char) - ord('가')
    jungsung_index = (code % (21 * 28)) // 28
    return JUNGSUNG[jungsung_index]

def get_jongsung(char):
    """ハングル文字から終声を抽出"""
    if not ('가' <= char <= '힣'):
        return None
    code = ord(char) - ord('가')
    jongsung_index = code % 28
    if jongsung_index == 0:
        return None  # 終声なし
    return JONGSUNG[jongsung_index - 1]


def match_jamo_pattern(word, jamo_pattern):
    """字母パターンにマッチするかチェック（完全版）"""
    if not word or not jamo_pattern:
        return False
    
    first_jamo = jamo_pattern[0]
    
    # 完成形ハングル → prefix一致
    if '\uAC00' <= first_jamo <= '\uD7A3':
        return word.startswith(jamo_pattern)
    
    # 初声のみ → 初声一致
    if first_jamo in CHOSUNG:
        if word and '\uAC00' <= word[0] <= '\uD7A3':
            return get_chosung(word[0]) == first_jamo
        return False
    
    # その他（アルファベット、数字など） → prefix一致（フォールバック）
    return word.startswith(jamo_pattern)


def replace_jamos_with_masks(line):
    """字母を完成形ハングルに変換してから[MASK]に置換し、位置情報を返す"""
    jamo_info = []
    result = []
    
    i = 0
    while i < len(line):
        char = line[i]
        converted = None
        jamo_len = 0
        
        # ★追加: カスタムマスクをスキップ
        if char == '[':
            # [MASK_X]形式を検出
            match = re.match(r'\[MASK[^\]]*\]', line[i:])
            if match:
                result.append(match.group())
                i += len(match.group())
                continue
        
        # 初声+中声+終声（3文字）
        if (i + 2 < len(line) and 
            line[i] in CHOSUNG and 
            line[i + 1] in JUNGSUNG and 
            line[i + 2] in JONGSUNG):
            cho_idx = CHOSUNG.index(line[i])
            jung_idx = JUNGSUNG.index(line[i + 1])
            jong_idx = JONGSUNG.index(line[i + 2]) + 1
            converted = chr(0xAC00 + cho_idx * 588 + jung_idx * 28 + jong_idx)
            jamo_len = 3
        
        # 初声+中声（2文字）
        elif i + 1 < len(line) and line[i] in CHOSUNG and line[i + 1] in JUNGSUNG:
            cho_idx = CHOSUNG.index(line[i])
            jung_idx = JUNGSUNG.index(line[i + 1])
            converted = chr(0xAC00 + cho_idx * 588 + jung_idx * 28)
            jamo_len = 2
        
        # 初声のみ（1文字）
        elif char in CHOSUNG:
            converted = char
            jamo_len = 1
        
        if converted:
            mask_index = len(jamo_info)
            jamo_info.append((mask_index, converted))
            result.append('[MASK]')
            i += jamo_len
        else:
            result.append(char)
            i += 1
    
    return ''.join(result), jamo_info
    
    
def count_consecutive_masks(text: str):
    """
    連続する [MASK] を検出し、
    [(開始文字位置, 連続個数), ...] を返す
    例:
      "[MASK]"            -> [(0, 1)]
      "[MASK][MASK]"      -> [(0, 2)]
      "a[MASK][MASK]b"    -> [(1, 2)]
    """
    results = []
    for m in re.finditer(r'(?:\[MASK\])+', text):
        count = m.group().count('[MASK]')
        results.append((m.start(), count))
    return results


class KoreanMaskCompletion:
    def __init__(self, model_path, top_k=50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
    
        kwargs = {}
        if "distilkobert" in model_path.lower():
            kwargs["trust_remote_code"] = True
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, **kwargs).to(self.device)
        self.model.eval()
        self.mask_token_id = self.tokenizer.mask_token_id
        self.top_k = top_k
        self.mask_token = self.tokenizer.mask_token
    
        # ★追加: Token IDキャッシュ構築
#       print("Building token ID cache...")
        self.id2token_cache = {
            i: self.tokenizer.convert_ids_to_tokens(i)
            for i in range(self.tokenizer.vocab_size)
        }
        self.unk_token = self.tokenizer.unk_token
#       print(f"Cached {len(self.id2token_cache)} tokens")
        
        # ★追加: ハングルトークンIDセットの事前構築
#       print("Building Hangul token filter...")
        self.hangul_token_ids = set()
    
        for token_id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
        
            # 特殊トークン除外
            if token_id in self.tokenizer.all_special_ids:
                continue
        
            # プレフィックス削除
            token_clean = token.lstrip('▁ĠĊ▃')
            if token_clean.startswith('##'):
                token_clean = token_clean[2:]
        
            # 空チェック
            if not token_clean:
                continue
        
            # ★修正: 完成形ハングルのみを含むトークンに限定
            if re.search(r'[\uAC00-\uD7A3]', token_clean):
                # 記号や数字、アルファベット、漢字が混ざっていないかチェック
                if re.fullmatch(r'[\uAC00-\uD7A3]+', token_clean):
                    has_jamo = any(ord(c) in range(0x1100, 0x1200) or ord(c) in range(0x3130, 0x3190) for c in token_clean)
                    if not has_jamo:
                        self.hangul_token_ids.add(token_id)
                    else:
                        print(f"[INIT] Filtered jamo in hangul_ids: {token_clean}")
    
        # TensorにしてGPUに転送
        hangul_ids_list = sorted(self.hangul_token_ids)
        self.hangul_ids_tensor = torch.tensor(hangul_ids_list, dtype=torch.long, device=self.device)
        
        print(f"Hangul tokens: {len(self.hangul_token_ids)}/{self.tokenizer.vocab_size} "
              f"({100*len(self.hangul_token_ids)/self.tokenizer.vocab_size:.1f}%)")
    
    @staticmethod
    def get_display_width(text):
        """文字列の表示幅を計算（全角=2、半角=1）"""
        width = 0
        for char in text:
            ea_width = unicodedata.east_asian_width(char)
            if ea_width in ('F', 'W'):
                width += 2  # 全角（Full/Wide）
            else:
                width += 1  # 半角（それ以外）
        return width
    
    def split_sentences(self, text):
        """テキストを文に分割"""
        return re.findall(r'[^.!?。！？]+[.!?。！？]?', text)
    
    def find_mask_sentence(self, sentences):
        """[MASK]を含む文を見つける"""
        for i, sent in enumerate(sentences):
            if '[MASK]' in sent:
                return i
        return -1
    
    def get_context_window(self, text, use_full_context=True):
        """文脈ウィンドウを取得（モデルの最大長を考慮）"""
        # モデルの最大長を動的に取得
        max_tokens = getattr(self.tokenizer, 'model_max_length', 512)
    
        # 異常値の場合のフォールバック
        if max_tokens > 100000:
            max_tokens = 512
    
        if not use_full_context:
            sentences = self.split_sentences(text)
            mask_idx = self.find_mask_sentence(sentences)
            if mask_idx >= 0:
                # [MASK]を含む文と前後1文ずつ（最大3文）
                start = max(0, mask_idx - 1)
                end = min(len(sentences), mask_idx + 2)
                return ''.join(sentences[start:end])
            return text

        # 既存のトークン数チェックコード
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= max_tokens:
            return text

        sentences = self.split_sentences(text)
        mask_idx = self.find_mask_sentence(sentences)

        if mask_idx < 0:
            decoded = self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            return decoded

        # [MASK]を含む文 + 直前1文 + 直後1文を優先
        context = [sentences[mask_idx]]

        # 直前の文を追加
        if mask_idx > 0:
            test = sentences[mask_idx - 1] + context[0]
            if len(self.tokenizer.encode(test, add_special_tokens=True)) <= max_tokens:
                context.insert(0, sentences[mask_idx - 1])

        # 直後の文を追加
        if mask_idx < len(sentences) - 1:
            test = ''.join(context) + sentences[mask_idx + 1]
            if len(self.tokenizer.encode(test, add_special_tokens=True)) <= max_tokens:
                context.append(sentences[mask_idx + 1])

        # まだ余裕があれば更に前後を追加
        left = mask_idx - 2 if mask_idx > 0 and len(context) > 1 else mask_idx - 1
        right = mask_idx + 2 if mask_idx < len(sentences) - 1 and len(context) > 1 else mask_idx + 1

        while left >= 0 or right < len(sentences):
            current = ''.join(context)
            current_tokens = len(self.tokenizer.encode(current, add_special_tokens=True))
    
            if current_tokens >= max_tokens:
                break
    
            if left >= 0 and (left != mask_idx - 1 or len(context) == 1):
                test = sentences[left] + current
                if len(self.tokenizer.encode(test, add_special_tokens=True)) <= max_tokens:
                    context.insert(0, sentences[left])
                    left -= 1
                else:
                    left = -1
    
            if right < len(sentences) and (right != mask_idx + 1 or len(context) == 1):
                test = current + sentences[right]
                if len(self.tokenizer.encode(test, add_special_tokens=True)) <= max_tokens:
                    context.append(sentences[right])
                    right += 1
                else:
                    right = len(sentences)

        return ''.join(context)
    
    
    
    def has_hangul(text):
        """完成形ハングル + 字母を検出"""
        return any(
            '\uAC00' <= c <= '\uD7A3' or  # 完成形ハングル（가-힣）
            '\u1100' <= c <= '\u11FF' or  # ハングル字母（初声・中声・終声）
            '\u3130' <= c <= '\u318F'     # 互換字母（ㄱ-ㅣ）
            for c in text
        )


    def predict_masks(self, text, jamo_info=None, use_full_context=False):
        """[MASK]トークンを予測"""
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)

        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        token_type_ids = None
        if 'token_type_ids' in inputs:
            token_type_ids = inputs['token_type_ids'].to(self.device)

        mask_token_indices = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_token_indices) == 0:
            return []

        try:
            import inspect
            forward_params = inspect.signature(self.model.forward).parameters
            if 'token_type_ids' not in forward_params:
                token_type_ids = None
        except:
            model_class = self.model.__class__.__name__.lower()
            if 'distil' in model_class or 'roberta' in model_class or 'electra' in model_class:
                token_type_ids = None
    
        with torch.no_grad():
            if token_type_ids is not None:
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits

        mask_blocks = []
        if len(mask_token_indices) > 0:
            start = mask_token_indices[0].item()
            length = 1
            for i in range(1, len(mask_token_indices)):
                if mask_token_indices[i].item() == mask_token_indices[i-1].item() + 1:
                    length += 1
                else:
                    mask_blocks.append((start, length))
                    start = mask_token_indices[i].item()
                    length = 1
            mask_blocks.append((start, length))

        jamo_map = dict(jamo_info) if jamo_info else {}

        all_results = []
        cursor = 0

        for start_pos, mask_len in mask_blocks:
            mask_logits_list = logits[0, mask_token_indices[cursor:cursor + mask_len]]
        
            topk_per_pos = []
            for mask_idx_position, mask_logits in enumerate(mask_logits_list):
                real_mask_idx = cursor + mask_idx_position
                jamo_pattern = jamo_map.get(real_mask_idx)
            
                # ★修正: 字母フィルタ用の取得数を調整
                top_k = min(200, self.tokenizer.vocab_size) if jamo_pattern else min(100, self.tokenizer.vocab_size)
            
                scaled_logits = mask_logits / TEMPERATURE
            
                # ★修正: ハングルフィルタをGPU上で実行
                if jamo_pattern:
                    # 字母あり: ハングルトークンのみに絞る
                    hangul_logits = scaled_logits[self.hangul_ids_tensor]
                    top_vals, top_indices = torch.topk(hangul_logits, min(top_k, len(self.hangul_ids_tensor)))
                    top_ids = self.hangul_ids_tensor[top_indices]
                else:
                    # ★修正: 字母なしでもハングルトークンのみに絞る
                    hangul_logits = scaled_logits[self.hangul_ids_tensor]
                    top_vals, top_indices = torch.topk(hangul_logits, min(top_k, len(self.hangul_ids_tensor)))
                    top_ids = self.hangul_ids_tensor[top_indices]
            
                top_probs = torch.softmax(top_vals, dim=0)
            
                tokens = []
                for i, p in zip(top_ids, top_probs):
                    token_id = i.item()
            
                    if token_id in self.tokenizer.all_special_ids:
                        continue
            
                    token = self.id2token_cache.get(token_id, self.unk_token)
                
                    token_clean = token.lstrip('▁ĠĊ▃')
                    if token_clean.startswith('##'):
                        token_clean = token_clean[2:]
            
                    if not token_clean:
                        continue



                    '''
                    # ★デバッグ: フィルタ動作確認
                    is_hangul = re.fullmatch(r'[가-힣]+', token_clean)
                    has_jamo = any(ord(c) in range(0x1100, 0x1200) or ord(c) in range(0x3130, 0x3190) for c in token_clean)

                    if not is_hangul:
                        print(f"[DEBUG] Filtered (not fullmatch): {token_clean}")
                        continue
                    if has_jamo:
                        print(f"[DEBUG] Filtered (has jamo): {token_clean}")
                        continue
                    '''


                    # ★修正: 純粋な完成形ハングルのみに制限（字母・記号・アルファベット除外）
                    # 完成形ハングル（가-힣）のみで構成されているかチェック
                    if not re.fullmatch(r'[가-힣]+', token_clean):
                        continue
                    # 字母（初声・中声・終声）が含まれていないか追加チェック
                    if any(ord(c) in range(0x1100, 0x1200) or ord(c) in range(0x3130, 0x3190) for c in token_clean):
                        continue
                        
                    # 字母フィルタリング
                    if jamo_pattern:
                        if not match_jamo_pattern(token_clean, jamo_pattern):
                            continue


            
                    tokens.append({
                        "token": token_clean,
                        "probability": p.item()
                    })
            
                # 確率正規化
                if tokens:
                    total_prob = sum(c["probability"] for c in tokens)
                    if total_prob > 0:
                        for c in tokens:
                            c["probability"] /= total_prob
        
                candidates = sorted(tokens, key=lambda x: x["probability"], reverse=True)
                candidates = candidates[:self.top_k]
        
                topk_per_pos.append({
                    "mask_index": real_mask_idx,
                    "candidates": candidates
                })
    
            all_results.append(topk_per_pos)
            cursor += mask_len

        flattened = []
        for block in all_results:
            flattened.extend(block)

        return flattened
    
    def evaluate_mlm_score_fast(self, text, skip_positions=None):
        if skip_positions is None:
            skip_positions = set()
        
        
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # モデルのforward引数チェック（フォールバック付き）
        try:
            import inspect
            forward_params = inspect.signature(self.model.forward).parameters
            inputs = {k: v for k, v in inputs.items() if k in forward_params}
        except Exception as e:
            model_class = self.model.__class__.__name__.lower()
            if 'distil' in model_class or 'roberta' in model_class or 'electra' in model_class:
                inputs.pop('token_type_ids', None)
        
        
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits
        
        # 各位置のスコア計算
        total_score = 0.0
        for idx, pos in enumerate(eval_positions):
            original_token = input_ids[0, pos].item()
            token_logits = logits[idx, pos]
            prob = torch.softmax(token_logits, dim=0)[original_token].item()
            total_score += math.log(prob + 1e-12)
        
        return total_score



def calculate_ensemble_scores_rank(all_model_predictions, model_names):
    """
    方式A: 順位ベーススコアリング（動的スコア計算）
    """
    if not model_names or not all_model_predictions:
        return []
    
    num_masks = len(all_model_predictions[model_names[0]])
    ensemble_results = []
    
    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions[model_name]
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                # 順位上限チェック
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                # 確率閾値チェック
                if prob < MIN_PROBABILITY_THRESHOLD:
                    break
                
                # 動的にスコアを計算
                score = get_rank_score(rank, RANK_SCORE_METHOD)
                
                if token not in token_scores:
                    token_scores[token] = {"score": 0, "models": []}
                
                token_scores[token]["score"] += score
                token_scores[token]["models"].append((model_name, rank, prob))
        
        sorted_tokens = sorted(
            token_scores.items(), 
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
    """
    方式B: 確率ベーススコアリング（変更なし）
    """
    if not model_names or not all_model_predictions:
        return []
    
    num_masks = len(all_model_predictions[model_names[0]])
    ensemble_results = []
    
    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions[model_name]
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                # 順位上限チェック
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                # 確率閾値チェック
                if prob < MIN_PROBABILITY_THRESHOLD:
                    break
                
                if token not in token_scores:
                    token_scores[token] = {"probs": [], "models": []}
                
                token_scores[token]["probs"].append(prob)
                token_scores[token]["models"].append((model_name, rank, prob))
        
        # 平均確率を計算
        for token in token_scores:
            avg_prob = sum(token_scores[token]["probs"]) / len(token_scores[token]["probs"])
            token_scores[token]["score"] = avg_prob
        
        sorted_tokens = sorted(
            token_scores.items(), 
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
    """
    方式C: ハイブリッドスコアリング（動的スコア計算）
    """
    if not model_names or not all_model_predictions:
        return []
    
    num_masks = len(all_model_predictions[model_names[0]])
    ensemble_results = []
    
    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions[model_name]
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                # 順位上限チェック
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                # 確率閾値チェック
                if prob < MIN_PROBABILITY_THRESHOLD:
                    break
                
                # 動的にスコアを計算
                rank_score = get_rank_score(rank, RANK_SCORE_METHOD)
                
                if token not in token_scores:
                    token_scores[token] = {"score": 0, "models": []}
                
                # 順位スコア × 確率
                hybrid_score = rank_score * prob
                token_scores[token]["score"] += hybrid_score
                token_scores[token]["models"].append((model_name, rank, prob))
        
        sorted_tokens = sorted(
            token_scores.items(), 
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


def save_ensemble_results(all_results, output_file_base, model_names, model_abbreviations, methods):
    """
    各方式のアンサンブル結果を別ファイルに保存
    """
    # スコア表を生成（説明用）
    score_table = "\n".join([
        f"  {i}位: {get_rank_score(i, RANK_SCORE_METHOD):.2f}点"
        for i in range(1, min(11, MAX_RANK_TO_CONSIDER + 1))
    ])
    
    method_configs = {
        "rank": {
            "name": "方式A: 順位ベース",
            "func": calculate_ensemble_scores_rank,
            "suffix": "_rank",
            "score_unit": "点",
            "score_format": ".2f",
            "description": f"順位スコア（{RANK_SCORE_METHOD}方式）:\n{score_table}\n確率閾値: {MIN_PROBABILITY_THRESHOLD*100:.1f}%以上, 評価範囲: 1-{MAX_RANK_TO_CONSIDER}位"
        },
        "probability": {
            "name": "方式B: 確率ベース",
            "func": calculate_ensemble_scores_probability,
            "suffix": "_probability",
            "score_unit": "",
            "score_format": ".4f",
            "description": f"各モデルの確率を平均\n確率閾値: {MIN_PROBABILITY_THRESHOLD*100:.1f}%以上, 評価範囲: 1-{MAX_RANK_TO_CONSIDER}位"
        },
        "hybrid": {
            "name": "方式C: ハイブリッド",
            "func": calculate_ensemble_scores_hybrid,
            "suffix": "_hybrid",
            "score_unit": "点",
            "score_format": ".3f",
            "description": f"順位スコア × 確率（{RANK_SCORE_METHOD}方式）:\n{score_table}\n確率閾値: {MIN_PROBABILITY_THRESHOLD*100:.1f}%以上, 評価範囲: 1-{MAX_RANK_TO_CONSIDER}位"
        }
    }
    
    for method in methods:
        if method not in method_configs:
            continue
        
        config = method_configs[method]
        output_file = output_file_base + config["suffix"] + ".txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"アンサンブル補完結果（{config['name']}）\n")
            f.write("=" * 80 + "\n")
            f.write(f"\n【スコアリング方式】\n")
            f.write(f"{config['description']}\n")
            f.write(f"参加モデル数: {len(model_names)}\n\n")
            
            for line_num, process_count, original_text, all_model_predictions, _ in all_results:
                f.write("=" * 80 + "\n")
                f.write(f"【行 {line_num}: {process_count}番目】\n")
                f.write(f"{original_text}\n")
                f.write("-" * 80 + "\n")
                
                ensemble_results = config["func"](all_model_predictions, model_names)
                
                if not ensemble_results:
                    f.write("候補なし\n\n")
                    continue

                
                for mask_idx, mask_result in enumerate(ensemble_results, 1):
                    f.write(f"\n[MASK]{mask_idx}:\n")

                    
                    if not mask_result:
                        f.write("  (条件に合致する候補が見つかりませんでした)\n")
                        continue
                    
                    for rank, result in enumerate(mask_result, 1):
                        token = result["token"]
                        score = result["score"]
                        support = result["support"]
                        
                        detail_parts = []
                        if result["details"]:
                            for model_name, model_rank, prob in result["details"]:
                                abbrev = model_abbreviations.get(model_name, model_name[:4])
                                detail_parts.append(f"{abbrev}:{model_rank:2d}位({prob:.4f})")
                        
                        detail_str = ", ".join(detail_parts)
                        
                        f.write(f" {rank:2d}位: {token}  ({score:{config['score_format']}}{config['score_unit']}) ← {support}モデルが支持（詳細: {detail_str}）\n")
                
                f.write("\n")
        
        print(f"{config['name']}の結果を {output_file} に保存しました")


def display_ensemble_results(all_model_predictions, model_names, model_abbreviations, methods, original_text, line_num=None, process_count=None):
    """
    各方式のアンサンブル結果を画面に表示
    """
    method_configs = {
        "rank": {
            "name": "方式A: 順位ベース",
            "func": calculate_ensemble_scores_rank,
            "score_unit": "点",
            "score_format": "5.2f"
        },
        "probability": {
            "name": "方式B: 確率ベース",
            "func": calculate_ensemble_scores_probability,
            "score_unit": "",
            "score_format": ".4f"
        },
        "hybrid": {
            "name": "方式C: ハイブリッド",
            "func": calculate_ensemble_scores_hybrid,
            "score_unit": "点",
            "score_format": "6.3f"
        }
    }
    
    for method in methods:
        if method not in method_configs:
            continue
        
        config = method_configs[method]
        
        print("\n" + "=" * 80)
        print(f"【アンサンブル結果: {config['name']}】")
        if line_num is not None and process_count is not None:
            print(f"行 {line_num}: {process_count}番目")
        print(f"（スコア方式: {RANK_SCORE_METHOD}, 確率閾値: {MIN_PROBABILITY_THRESHOLD*100:.1f}%以上, 評価範囲: 1-{MAX_RANK_TO_CONSIDER}位）")
        print(original_text)
        print("-" * 80)
        
        ensemble_results = config["func"](all_model_predictions, model_names)
        
        if not ensemble_results:
            print("候補なし")
            continue
        
        for mask_idx, mask_result in enumerate(ensemble_results, 1):
            print(f"\n[MASK]{mask_idx}:")
            
            
            if not mask_result:
                print("  候補なし")
                continue
            
            for rank, result in enumerate(mask_result, 1):
                token = result["token"]
                score = result["score"]
                support = result["support"]
                
                detail_parts = []
                if result["details"]:
                    for model_name, model_rank, prob in result["details"]:
                        abbrev = model_abbreviations.get(model_name, model_name[:4])
                        detail_parts.append(f"{abbrev}:{model_rank:2d}位({prob:.4f})")
                
                detail_str = ", ".join(detail_parts)
                
                print(f" {rank:2d}位: {token}  ({score:{config['score_format']}}{config['score_unit']}) ← {support}モデルが支持（詳細: {detail_str}）")




def display_comparison_results(all_model_predictions, model_names, line_text, line_num=None, process_count=None, show_prob=True, decimals=4):
    """複数モデルの結果を比較表示"""
    print("\n" + "="*80)
    if line_num is not None and process_count is not None:
        print(f"行 {line_num}: {process_count}番目")
    print(line_text)
    print("-"*80)
    
    
    
    # 各[MASK]ごとに処理
    num_masks = len(all_model_predictions[model_names[0]]) if model_names and all_model_predictions[model_names[0]] else 0
    
    # 各モデルの最大セル幅を計算
    max_widths = []
    for model_name in model_names:
        max_width = len(MODEL_ABBREVIATIONS.get(model_name, model_name[:4]))
        for mask_idx in range(num_masks):
            predictions = all_model_predictions[model_name]
            if mask_idx < len(predictions):
                for rank in range(TOP_K):
                    if rank < len(predictions[mask_idx]['candidates']):
                        cand = predictions[mask_idx]['candidates'][rank]
                        if show_prob:
                            cell = f"{cand['token']} ({cand['probability']:.{decimals}f})"
                        else:
                            cell = cand['token']
                        width = KoreanMaskCompletion.get_display_width(cell)
                        max_width = max(max_width, width)
        max_widths.append(max_width + 2)  # マージン
    
    for mask_idx in range(num_masks):
        if mask_idx > 0:
            print()
        
        print(f"[MASK]{mask_idx + 1}:")
        
        # ヘッダー行（モデル名）
        header = "      "
        for model_name, col_width in zip(model_names, max_widths):
            abbrev = MODEL_ABBREVIATIONS.get(model_name, model_name[:4])
            abbrev_width = KoreanMaskCompletion.get_display_width(abbrev)
            padding = col_width - abbrev_width
            header += abbrev + " " * padding
        print(header)
        
        # 各順位ごとに表示
        for rank in range(TOP_K_DISPLAY):
            rank_label = f"{rank+1:2d}位:"
            line = f"{rank_label}  "
            for model_name, col_width in zip(model_names, max_widths):
                predictions = all_model_predictions[model_name]
                if mask_idx < len(predictions) and rank < len(predictions[mask_idx]['candidates']):
                    cand = predictions[mask_idx]['candidates'][rank]
                    if show_prob:
                        cell = f"{cand['token']} ({cand['probability']:.{decimals}f})"
                    else:
                        cell = cand['token']
                    cell_width = KoreanMaskCompletion.get_display_width(cell)
                    padding = col_width - cell_width
                    line += cell + " " * padding
                else:
                    line += " " * col_width
            print(line)

def save_comparison_to_txt(all_results, output_file, show_prob=True, decimals=6):
    """比較結果をTSV形式で保存"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for line_num, process_count, text, all_model_predictions, model_names in all_results:
            f.write("=" * 80 + "\n")
            f.write(f"行 {line_num}: {process_count}番目\n")
            f.write("-" * 80 + "\n")
            f.write(text + "\n")
            f.write("-" * 80 + "\n")
            
            # ★修正: 実際のマスク数を取得
            num_masks = len(next(iter(all_model_predictions.values()), []))
            
            for mask_idx in range(num_masks):
                f.write(f"[MASK]{mask_idx + 1}:\n")

                # ヘッダー
                header = [""] + [
                    MODEL_ABBREVIATIONS.get(m, m[:4]) for m in model_names
                ]
                f.write("\t".join(header) + "\n")

                # ===== 各モデルの最大セル幅を事前計算 =====
                max_widths = []
                for m in model_names:
                    max_width = len(MODEL_ABBREVIATIONS.get(m, m[:4]))
                    preds = all_model_predictions.get(m, [])
                    if mask_idx < len(preds):
                        for rank in range(TOP_K_DISPLAY):
                            if rank < len(preds[mask_idx]['candidates']):
                                cand = preds[mask_idx]['candidates'][rank]
                                w = cand['token']
                                s = cand['probability']
                                cell = f"{w} ({s:.{decimals}f})" if show_prob else w
                                width = KoreanMaskCompletion.get_display_width(cell)
                                max_width = max(max_width, width)
                    max_widths.append(max_width)

                # ===== 出力（空白で列位置を揃える）=====
                for rank in range(TOP_K_DISPLAY):
                    row = [f"{rank+1}位:"]
                    for i, m in enumerate(model_names):
                        preds = all_model_predictions.get(m, [])
                        if mask_idx < len(preds) and rank < len(preds[mask_idx]['candidates']):
                            cand = preds[mask_idx]['candidates'][rank]
                            w = cand['token']
                            s = cand['probability']
                            cell = f"{w} ({s:.{decimals}f})" if show_prob else w
                        else:
                            cell = ""
                        
                        # 列幅に合わせてパディング
                        cell_width = KoreanMaskCompletion.get_display_width(cell)
                        padding = max_widths[i] - cell_width
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
    output_file = input_path.parent / (input_path.stem + "_out" + input_path.suffix)
    
    print(f"入力ファイル: {Path(input_file).absolute()}")
    print(f"出力ファイル: {output_file.absolute()}")
    
    # モデルをロード
    models = {}
    for model_name in COMPARE_MODELS:
        model_path = f"models/{model_name.replace('/', '_')}"
        if not Path(model_path).exists():
            print(f"警告: モデルが見つかりません: {model_path} (スキップします)")
            continue
        models[model_name] = KoreanMaskCompletion(model_path, top_k=TOP_K)
    
    if not models:
        print("エラー: 使用可能なモデルがありません")
        return
    
    # 入力ファイル読み込み
    if not Path(input_file).exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_results = []
    
    # ===== ステップ1: 全行を読み込み、マスクタイプごとに分類 =====
    mask_groups = {}  # {'MASK': [...], 'MASK_A': [...], 'MASK_BAC': [...], ...}
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        '''
        # 字母モード処理（文脈を考慮しない）
        original_line = line
        jamo_info = []
        if JAMO_MODE and any(c in line for c in CHOSUNG + JUNGSUNG + ' '):
            line, jamo_info = replace_jamos_with_masks(line)
            
            # 字母が検出された場合は常に文脈削除
            if jamo_info:
                line = '[MASK]'
                print(f"字母検出 → 문맥 삭제: '{original_line}' → '{line}'")
        '''
        # 字母モード処理
        original_line = line
        jamo_info = []
        if JAMO_MODE and any(c in line for c in CHOSUNG + JUNGSUNG + ' '):
            line, jamo_info = replace_jamos_with_masks(line)
            
            if jamo_info:
                print(f"字母検出 → 완성형 変換 後 処理: '{original_line}' → '{line}'")
        
        # ★削除: この部分は後で処理する
        # results = models[model_name].predict_masks(...)



        
        
        # 【重要】複数[MASK]がある場合、個別エントリーに分割
        mask_count = line.count('[MASK]')
        
        if mask_count > 1 and jamo_info:
            for idx, (pos, jamo) in enumerate(jamo_info):
                temp_line = '[MASK]'
        
                mask_groups['MASK'].append({
                    'line_num': line_num,
                    'original': original_line,
                    'processed': temp_line,
                    'jamo_info': [(0, jamo)],  # ★修正: 常に0（[MASK]は1つだけ）
                    'mask_type': '[MASK]',
                    'mask_index': idx
                })
                
                
            continue
        
        # カスタムマスク（[MASK_XXX]）を検出
        custom_masks = re.findall(r'\[MASK_[A-Z0-9_]+\]', line)
        
        if custom_masks:
            # 各カスタムマスクタイプごとに登録
            for mask_type in set(custom_masks):
                mask_key = mask_type[1:-1]  # "[MASK_A]" -> "MASK_A"
                
                # この行に複数の同じカスタムマスクがある場合
                mask_count_in_line = line.count(mask_type)
                
                if mask_count_in_line == 1:
                    # 1つのみ → 通常登録
                    if mask_key not in mask_groups:
                        mask_groups[mask_key] = []
                    mask_groups[mask_key].append({
                        'line_num': line_num,
                        'original': original_line,
                        'processed': line,
                        'jamo_info': jamo_info,
                        'mask_type': mask_type
                    })
                else:
                    # 複数ある → 各[MASK_X]を個別エントリーとして登録
                    for idx in range(mask_count_in_line):
                        # idx番目の[MASK_X]のみ残し、他をプレースホルダーに
                        temp_line = line
                        mask_positions = [m.start() for m in re.finditer(re.escape(mask_type), line)]
                        
                        offset = 0
                        for i, pos in enumerate(mask_positions):
                            if i == idx:
                                continue
                            adjusted_pos = pos + offset
                            temp_line = temp_line[:adjusted_pos] + '__PLACEHOLDER__' + temp_line[adjusted_pos+len(mask_type):]
                            offset += len('__PLACEHOLDER__') - len(mask_type)
                        
                        if mask_key not in mask_groups:
                            mask_groups[mask_key] = []
                        mask_groups[mask_key].append({
                            'line_num': line_num,
                            'original': original_line,
                            'processed': temp_line,
                            'jamo_info': jamo_info if idx == 0 else [],
                            'mask_type': mask_type,
                            'mask_index': idx
                        })
                        
        # 通常の[MASK]の処理
        if '[MASK]' in line:
            mask_count = line.count('[MASK]')
            
            if mask_count == 1:
                # 1つの[MASK] → 通常処理
                if 'MASK' not in mask_groups:
                    mask_groups['MASK'] = []
                mask_groups['MASK'].append({
                    'line_num': line_num,
                    'original': original_line,
                    'processed': line,
                    'jamo_info': jamo_info,
                    'mask_type': '[MASK]'
                })
            else:
                # 複数の[MASK] → 各[MASK]を個別エントリーとして登録
                for idx in range(mask_count):
                    # この行のidx番目の[MASK]のみを処理対象とする
                    temp_line = line
                    
                    # 他の[MASK]を一時的にプレースホルダーに置換
                    mask_positions = []
                    for m in re.finditer(r'\[MASK\]', line):
                        mask_positions.append(m.start())
                    
                    # idx番目以外の[MASK]を`__PLACEHOLDER__`に置換
                    offset = 0
                    for i, pos in enumerate(mask_positions):
                        if i == idx:
                            continue
                        adjusted_pos = pos + offset
                        temp_line = temp_line[:adjusted_pos] + '__PLACEHOLDER__' + temp_line[adjusted_pos+6:]
                        offset += len('__PLACEHOLDER__') - 6
                    
                    if 'MASK' not in mask_groups:
                        mask_groups['MASK'] = []
                    
                    mask_groups['MASK'].append({
                        'line_num': line_num,
                        'original': original_line,
                        'processed': temp_line,
                        'jamo_info': jamo_info if idx == 0 else [],  # 字母情報は最初の[MASK]のみ
                        'mask_type': '[MASK]',
                        'mask_index': idx
                    })
    
    

    
    
    # ===== ステップ2: 各マスクグループを処理 =====
    process_count = 0
    
    for mask_key, items in mask_groups.items():
        if mask_key == 'MASK':
            # 通常の[MASK]: 個別に処理
            for item in items:
                process_count += 1
                line_num = item['line_num']
                original_line = item['original']
                line = item['processed']
                jamo_info = item['jamo_info']
                
                print(f"\n処理中: 行 {line_num}: {process_count}番目 [MASK]")
                if jamo_info:
                    print(f"字母検出: {jamo_info}")
                
                all_model_predictions = {}
                
                for model_name, model in models.items():
                    preds = model.predict_masks(
                        line, 
                        use_full_context=USE_FULL_LINE_CONTEXT,
                        jamo_info=jamo_info  # ★追加: jamo_infoを渡す
                    )
                    
                    # ★削除: predict_masks内で既にフィルタ済み
                    # if jamo_info:
                    #     ...（字母フィルタリングのコードを全て削除）
                    
                    all_model_predictions[model_name] = preds
                
                display_comparison_results(all_model_predictions, list(models.keys()), original_line,
                                           line_num=line_num, process_count=process_count,
                                           show_prob=SHOW_PROBABILITY, decimals=PROB_DECIMALS_DISPLAY)
                
                display_ensemble_results(all_model_predictions, list(models.keys()), 
                                        MODEL_ABBREVIATIONS, ENSEMBLE_METHODS, original_line,
                                        line_num=line_num, process_count=process_count)
                
                all_results.append((line_num, process_count, original_line, all_model_predictions, list(models.keys())))
        
        else:
            # カスタムマスク（[MASK_X]）: 連結して候補生成 + 文ペア評価
            process_count += 1
            
            # 準備
            combined_lines = []
            line_nums = []
            original_lines = []
            
            for item in items:
                # カスタムマスクを[MASK]に置換
                temp_line = item['processed'].replace(item['mask_type'], '[MASK]')
                combined_lines.append(temp_line)
                line_nums.append(item['line_num'])
                original_lines.append(item['original'])
            
            # 512トークンチェック
            first_model = list(models.values())[0]
            combined_text = first_model.tokenizer.sep_token.join(combined_lines)

#            for i, e in enumerate(entries):
#                print(f"  Entry{i}: '{e['processed']}', jamo_info={e['jamo_info']}")            
            
            tokens = first_model.tokenizer.encode(combined_text, add_special_tokens=True)
            
            if len(tokens) > MAX_TOKENS:
                print(f"\n警告: {mask_key}のグループが512トークンを超えています（{len(tokens)}トークン）")
                print(f"個別に処理します")
                
                # 個別処理
                for item in items:
                    line_num = item['line_num']
                    original_line = item['original']
                    line = item['processed'].replace(item['mask_type'], '[MASK]')
                    jamo_info = item['jamo_info']
                    
                    print(f"\n処理中: 行 {line_num}: {process_count}番目 {mask_key} (個別)")
                    
                    all_model_predictions = {}
                    
                    for model_name, model in models.items():
                        preds = model.predict_masks(line, use_full_context=USE_FULL_LINE_CONTEXT)
                        
                        if jamo_info:
                            for pred_idx, (_, jamos) in enumerate(jamo_info):
                                if pred_idx < len(preds):
                                    filtered_cands = [
                                        c for c in preds[pred_idx]['candidates']
                                        if len(c['token']) > 0 and get_chosung(c['token'][0]) == jamos[0]
                                    ]
                                    preds[pred_idx]['candidates'] = filtered_cands
                        
                        all_model_predictions[model_name] = preds
                    
                    display_comparison_results(all_model_predictions, list(models.keys()), original_line,
                                               line_num=line_num, process_count=process_count,
                                               show_prob=SHOW_PROBABILITY, decimals=PROB_DECIMALS_DISPLAY)
                    
                    display_ensemble_results(all_model_predictions, list(models.keys()), 
                                            MODEL_ABBREVIATIONS, ENSEMBLE_METHODS, original_line,
                                            line_num=line_num, process_count=process_count)
                    
                    all_results.append((line_num, process_count, original_line, all_model_predictions, list(models.keys())))
                
                continue
            
            
            # ===== 候補生成 + 文ペア評価 =====
            print(f"\n処理中: {mask_key}グループ（{len(items)}行連結）: {process_count}番目")
            print(f"対象行: {', '.join([f'行{ln}' for ln in line_nums])}")
            
            all_model_predictions = {}
            
            
            
            # 文ペア評価用モデル
            PAIR_EVAL_MODELS = {
                'klue/roberta-large',
                'klue/roberta-base',
                'klue/bert-base'
            }
            
            for model_name, model in models.items():
                # 信頼できるモデル以外はスキップ
                if model_name not in PAIR_EVAL_MODELS:
                    preds = model.predict_masks(combined_text, use_full_context=USE_FULL_LINE_CONTEXT)
                    all_model_predictions[model_name] = preds
                    continue
                
                # Step1: 各文を個別に候補生成
                individual_preds = []
                for text in combined_lines:
                    p = model.predict_masks(text, use_full_context=USE_FULL_LINE_CONTEXT)
                    if p:
                        individual_preds.append(p[0])
                
                if len(individual_preds) < 2:
                    preds = model.predict_masks(combined_text, use_full_context=USE_FULL_LINE_CONTEXT)
                    all_model_predictions[model_name] = preds
                    continue
                
                # Step2: 候補プール + 支持数フィルタ
                cand_maps = []
                for pred in individual_preds:
                    cand_map = {
                        c['token']: c['probability']
                        for c in pred['candidates'][:CANDIDATE_POOL_TOP_K]
                    }
                    cand_maps.append(cand_map)
                
                all_tokens = set()
                for m in cand_maps:
                    all_tokens |= set(m.keys())
                
                token_support = {}
                for token in all_tokens:
                    support_count = sum(1 for m in cand_maps if token in m)
                    token_support[token] = support_count
                
                candidate_pool = set()
                for token in all_tokens:
                    if token_support[token] < 2:
                        continue
                    if not re.fullmatch(r'[가-힣]+', token):
                        continue
                    candidate_pool.add(token)
                
                if not candidate_pool:
                    preds = model.predict_masks(combined_text, use_full_context=USE_FULL_LINE_CONTEXT)
                    all_model_predictions[model_name] = preds
                    continue
                
                # Step3: 文ペア評価
                rescored = []
                temp_combined = model.tokenizer.sep_token.join(combined_lines)
                inputs = model.tokenizer(temp_combined, return_tensors="pt", add_special_tokens=True)
                mask_token_positions = (inputs["input_ids"] == model.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist()
                mask_positions_set = set(mask_token_positions)
                
                for token in candidate_pool:
                    filled = []
                    for line in combined_lines:
                        filled.append(line.replace('[MASK]', token, 1))
                    
                    filled_text = model.tokenizer.sep_token.join(filled)
                    pll_score = model.evaluate_mlm_score_fast(filled_text, skip_positions=mask_positions_set)
                    
                    support_weight = 1 + (token_support[token] - 1) * 0.7
                    final_score = pll_score * support_weight
                    
                    rescored.append({
                        'token': token,
                        'probability': final_score
                    })
                
                # ソート + Softmax正規化
                rescored.sort(key=lambda x: -x['probability'])
                
                if rescored:
                    max_log_prob = max(x['probability'] for x in rescored)
                    exp_scores = [math.exp(x['probability'] - max_log_prob) for x in rescored]
                    total = sum(exp_scores)
                    for i, item in enumerate(rescored):
                        item['probability'] = exp_scores[i] / total if total > 0 else 0.0
                
                all_model_predictions[model_name] = [{
                    'position': 0,
                    'candidates': rescored[:TOP_K]
                }]

                
            
            # 表示用テキスト
            display_text = f"{mask_key}グループ:\n" + "\n".join([f"  行{ln}: {ol}" for ln, ol in zip(line_nums, original_lines)])
            
            display_comparison_results(all_model_predictions, list(models.keys()), display_text,
                                       line_num=line_nums[0], process_count=process_count,
                                       show_prob=SHOW_PROBABILITY, decimals=PROB_DECIMALS_DISPLAY)
            
            display_ensemble_results(all_model_predictions, list(models.keys()), 
                                    MODEL_ABBREVIATIONS, ENSEMBLE_METHODS, display_text,
                                    line_num=line_nums[0], process_count=process_count)
            
            all_results.append((line_nums[0], process_count, display_text, all_model_predictions, list(models.keys())))
    
    # TXT保存
    try:
        save_comparison_to_txt(all_results, output_file, show_prob=SHOW_PROBABILITY, decimals=PROB_DECIMALS_DISPLAY)
        print(f"\n結果を {output_file} に保存しました")
        print(f"ファイルサイズ: {Path(output_file).stat().st_size} バイト")
        
        # アンサンブル結果の保存（3方式全て）
        ensemble_file_base = str(input_path.parent / (input_path.stem + "_ensemble"))
        save_ensemble_results(all_results, ensemble_file_base, list(models.keys()), 
                             MODEL_ABBREVIATIONS, ENSEMBLE_METHODS)
        print(f"\n通常結果を {output_file} に保存しました")
    except Exception as e:
        print(f"\nエラー: ファイル保存失敗 - {e}")

if __name__ == "__main__":
    main()