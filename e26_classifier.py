from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import json
import math
import re
import os

# =========================
# 1. 데이터 구조 정의
# =========================

@dataclass
class EquipmentInput:
    """장비 입력 정보 (Asset Inventory 한 줄이라고 보면 됨)"""
    id: str
    name: str                      # e.g. "Main Engine Remote Control System"
    vendor: str = ""
    system: str = ""               # e.g. "Main Engine", "Cargo Control", "BNWAS"
    function_hint: Optional[str] = None  # 사용자가 넣는 Function 힌트 (예: "Navigation")
    description: Optional[str] = None    # 자유 입력 설명
    is_computer_based: Optional[bool] = None
    controls_physical_process: Optional[bool] = None
    network_zone: Optional[str] = None   # 예: "Navigation", "Cargo", "Admin", "Crew"
    connected_to_external_network: Optional[bool] = None


@dataclass
class E26Decision:
    is_cbs: bool
    function_class: Optional[str]
    in_scope: bool
    criteria_matches: Dict[str, bool]
    score: float
    confidence: float
    warnings: List[str] = field(default_factory=list)
    explanation: str = ""


# =========================
# 2. E26 룰 정의
# =========================
# UR E26 1.3.2 및 DNV/ABS 요약 리스트를 기반으로 한 카테고리 구성 (문구는 재구성, 구조는 사실 기반)

E26_FUNCTION_RULES: List[Dict] = [
    # 1) Propulsion
    {
        "id": "PROPULSION",
        "name_en": "Propulsion control & monitoring",
        "name_kr": "추진 제어 및 감시",
        "keywords": [
            "main engine", "me control", "propulsion", "cpp", "controllable pitch",
            "thruster", "azimuth", "azipod", "shaft generator", "m/e remote",
            "engine control", "main propulsion", "me rc", "propulsion control"
        ],
        "default_criticality": {
            "safety": True,
            "environment": True,
            "operation": True
        }
    },

    # 2) Steering
    {
        "id": "STEERING",
        "name_en": "Steering control",
        "name_kr": "조타 제어",
        "keywords": [
            "steering", "rudder", "steering gear", "autopilot", "steering control",
            "steering system"
        ],
        "default_criticality": {
            "safety": True,
            "environment": False,
            "operation": True
        }
    },

    # 3) Anchoring & Mooring
    {
        "id": "ANCHORING_MOORING",
        "name_en": "Anchoring and mooring control",
        "name_kr": "계류 및 닻 장비 제어",
        "keywords": [
            "windlass", "mooring", "mooring winch", "capstan", "anchor winch",
            "anchoring", "mooring control"
        ],
        "default_criticality": {
            "safety": True,
            "environment": False,
            "operation": True
        }
    },

    # 4) Power generation & distribution
    {
        "id": "POWER",
        "name_en": "Power generation and distribution",
        "name_kr": "전력 생산 및 분배",
        "keywords": [
            "msb", "main switchboard", "switchboard", "emergency switchboard",
            "esb", "power management", "pms", "generator control", "synchro",
            "diesel generator", "dg control", "bus-tie", "bus tie", "ups system"
        ],
        "default_criticality": {
            "safety": True,
            "environment": True,
            "operation": True
        }
    },

    # 5) Fire detection & extinguishing
    {
        "id": "FIRE_SAFETY",
        "name_en": "Fire detection and extinguishing",
        "name_kr": "화재 감지 및 소화 시스템",
        "keywords": [
            "fire detection", "fire alarm", "fds", "fdas", "fire & gas", "f&g",
            "foam system", "co2 system", "water mist", "sprinkler", "fixed fire",
            "fire control panel", "fire safety"
        ],
        "default_criticality": {
            "safety": True,
            "environment": True,
            "operation": True
        }
    },

    # 6) Bilge, ballast, loading computer
    {
        "id": "BILGE_BALLAST_LOADING",
        "name_en": "Bilge, ballast and loading computer",
        "name_kr": "빌지·밸러스트 및 로딩 컴퓨터",
        "keywords": [
            "ballast", "bilge", "bwms", "ballast water management",
            "loading computer", "cargo loading computer", "stability computer",
            "draft calculation", "trim & stability"
        ],
        "default_criticality": {
            "safety": True,
            "environment": True,
            "operation": True
        }
    },

    # 7) Watertight integrity & flooding detection
    {
        "id": "WATERTIGHT_FLOODING",
        "name_en": "Watertight integrity and flooding detection",
        "name_kr": "수밀 및 침수 감시",
        "keywords": [
            "watertight door", "watertight", "flooding detection", "flood alarm",
            "hull integrity monitoring", "draft monitoring", "tank level alarm",
            "remote closing", "remote operated valve control"
        ],
        "default_criticality": {
            "safety": True,
            "environment": True,
            "operation": True
        }
    },

    # 8) Lighting (emergency, navigation lights, etc.)
    {
        "id": "LIGHTING",
        "name_en": "Lighting and signalling",
        "name_kr": "비상/항해 조명 및 신호등",
        "keywords": [
            "emergency lighting", "emergency light", "navigation light",
            "nav light", "signal light", "lll", "low location lighting",
            "lighting control", "illumination"
        ],
        "default_criticality": {
            "safety": True,
            "environment": False,
            "operation": True
        }
    },

    # 9) Navigation
    {
        "id": "NAVIGATION",
        "name_en": "Navigation systems",
        "name_kr": "항해 시스템",
        "keywords": [
            "ecs", "ecdis", "gps", "gnss", "radar", "ais", "bnwas",
            "speed log", "speedlog", "gyro", "gyrocompass", "navtex",
            "conning", "integrated bridge", "ibs", "ibridge", "track control",
            "navigation system", "voyage data recorder", "vdr"
        ],
        "default_criticality": {
            "safety": True,
            "environment": False,
            "operation": True
        }
    },

    # 10) Radiocommunication / GMDSS
    {
        "id": "RADIOCOMM",
        "name_en": "Radiocommunication and distress",
        "name_kr": "무선통신 및 조난 통신",
        "keywords": [
            "gmdss", "vhf", "mf/hf", "ssb", "inmarsat", "satcom",
            "fleet broadband", "fb500", "cobham", "epirb", "ssas",
            "sart", "radio console", "radio equipment", "distress alarm"
        ],
        "default_criticality": {
            "safety": True,
            "environment": False,
            "operation": True
        }
    },

    # 11) HVAC / Ventilation
    {
        "id": "HVAC_VENTILATION",
        "name_en": "HVAC and ventilation",
        "name_kr": "공조 및 환기 제어",
        "keywords": [
            "hvac", "ventilation", "engine room fan", "accommodation fan",
            "fan control", "smoke damper", "ventilation control", "air handling",
            "ahu", "air handling unit"
        ],
        "default_criticality": {
            "safety": True,      # 연기 제어/환기 측면에서 안전 영향 가능 → True (추측입니다)
            "environment": False,
            "operation": True
        }
    },

    # 12) Central automation / IAS / ICMS
    {
        "id": "AUTOMATION_IAS",
        "name_en": "Integrated automation and monitoring",
        "name_kr": "통합 자동화·감시 시스템",
        "keywords": [
            "ias", "integrated automation", "icms", "iamcs", "ams",
            "ship automation", "machinery monitoring", "power management",
            "ems", "engine monitoring system", "central monitoring"
        ],
        "default_criticality": {
            "safety": True,
            "environment": True,
            "operation": True
        }
    },

    # 13) Security & access control / CCTV / PA/GA
    {
        "id": "SECURITY_ACCESS",
        "name_en": "Security, access control and surveillance",
        "name_kr": "보안·출입 통제·감시",
        "keywords": [
            "cctv", "access control", "door control", "video surveillance",
            "security system", "pa/ga", "public address", "general alarm",
            "ga system", "security monitoring"
        ],
        "default_criticality": {
            "safety": True,      # PA/GA 포함 고려 (조기 경보 역할, 추측입니다)
            "environment": False,
            "operation": True
        }
    },

    # 14) 기타 필수 시스템 버킷 (룰에 안 걸리는 것용 fallback)
    {
        "id": "OTHER_ESSENTIAL",
        "name_en": "Other essential ship systems",
        "name_kr": "기타 필수 선박 시스템",
        "keywords": [
            # 의도: 룰로 명확히 매칭되지 않는 'automation', 'control system' 이 들어왔을 때
            "control system", "monitoring system", "safety system",
            "emergency shutdown", "esd system", "alarm system",
            "process control", "automation system"
        ],
        "default_criticality": {
            "safety": True,
            "environment": True,
            "operation": True
        }
    },
]

# Criteria 이름은 회사/프로젝트에서 합의해서 쓰면 됨
CRITERIA_RULES: Dict[str, Dict] = {
    "safety_critical": {
        "description": "Failure could lead to unsafe condition for human life or ship",
    },
    "environment_critical": {
        "description": "Failure could lead to pollution or environmental damage",
    },
    "operation_critical": {
        "description": "Essential for ship operation / commercial service",
    },
    "networked_with_other_cbs": {
        "description": "Connected to other CBS via onboard networks",
    },
    "connected_to_untrusted": {
        "description": "Directly or indirectly connected to untrusted networks (e.g. internet, crew wifi)",
    },
}


# =========================
# 3. 단순 Naive Bayes 텍스트 모델
# =========================

class NaiveBayesTextModel:
    """
    아주 단순한 텍스트 분류기 (Function Class 용도)
    - 토큰: 장비 이름, 시스템, 설명, 벤더의 알파벳/숫자 단어
    - 라벨: Function Class ID (예: "PROPULSION", "NAVIGATION")
    """
    def __init__(self):
        self.class_counts = Counter()
        self.token_counts = defaultdict(Counter)
        self.vocab = set()
        self.total_samples = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def update(self, label: str, text: str):
        tokens = self.tokenize(text)
        if not tokens:
            return
        self.class_counts[label] += 1
        self.total_samples += 1
        for t in tokens:
            self.token_counts[label][t] += 1
            self.vocab.add(t)

    def predict_proba(self, text: str) -> Dict[str, float]:
        tokens = self.tokenize(text)
        if not tokens or not self.total_samples:
            return {}

        vocab_size = len(self.vocab) if self.vocab else 1
        scores = {}
        for label in self.class_counts:
            # log P(label)
            log_prob = math.log(self.class_counts[label] / self.total_samples)
            total_tokens_in_class = sum(self.token_counts[label].values()) + vocab_size

            for t in tokens:
                # Laplace smoothing
                token_count = self.token_counts[label][t] + 1
                log_prob += math.log(token_count / total_tokens_in_class)

            scores[label] = log_prob

        # normalize to probabilities
        max_log = max(scores.values())
        exp_sum = sum(math.exp(v - max_log) for v in scores.values())
        probs = {lbl: math.exp(v - max_log) / exp_sum for lbl, v in scores.items()}
        return probs

    def to_json(self) -> Dict:
        return {
            "class_counts": dict(self.class_counts),
            "token_counts": {cls: dict(cnt) for cls, cnt in self.token_counts.items()},
            "vocab": list(self.vocab),
            "total_samples": self.total_samples,
        }

    @classmethod
    def from_json(cls, data: Dict) -> "NaiveBayesTextModel":
        model = cls()
        model.class_counts = Counter(data.get("class_counts", {}))
        model.token_counts = defaultdict(Counter)
        for cls_label, cnts in data.get("token_counts", {}).items():
            model.token_counts[cls_label] = Counter(cnts)
        model.vocab = set(data.get("vocab", []))
        model.total_samples = data.get("total_samples", 0)
        return model


# =========================
# 4. 메인 분류기
# =========================

class E26Classifier:
    def __init__(self, model_path: str = "e26_nb_model.json"):
        self.model_path = model_path
        self.nb_model = NaiveBayesTextModel()
        self._load_model()

    # ----- model persistence -----

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.nb_model = NaiveBayesTextModel.from_json(data)
            except Exception as e:
                print(f"[WARN] Failed to load model: {e}")

    def save_model(self):
        data = self.nb_model.to_json()
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ----- helper methods -----

    @staticmethod
    def _combine_text(e: EquipmentInput) -> str:
        fields = [
            e.name or "",
            e.vendor or "",
            e.system or "",
            e.description or "",
            e.function_hint or "",
        ]
        return " ".join(fields)

    @staticmethod
    def _is_cbs_by_heuristic(e: EquipmentInput) -> bool:
        """
        UR E26 정의에 따라, OT CBS 추정용 간단한 휴리스틱.
        - 확실한 부분: UR E26은 '물리 프로세스를 제어/감시하는 컴퓨터 기반 시스템'을 CBS로 봄.
        - 아래 조건/키워드는 현재 단계에선 실무 감각 기반 추정(추측입니다).
        """
        if e.is_computer_based is True:
            return True
        if e.is_computer_based is False:
            return False

        text = (e.name + " " + (e.system or "") + " " + (e.description or "")).lower()
        # 컴퓨터 기반일 확률이 높은 키워드
        cbs_keywords = [
            "system", "controller", "plc", "hmi", "server",
            "workstation", "control", "monitoring", "automation",
            "ipc", "panel pc", "gateway"
        ]
        for kw in cbs_keywords:
            if kw in text:
                return True

        # 반대로 단순 패시브 장비일 가능성이 높은 경우
        passive_keywords = [
            "valve", "pump", "sensor", "transmitter", "motor", "cabinet"
        ]
        if any(kw in text for kw in passive_keywords):
            return False

        # 정보 부족시 보수적으로 False
        return False

    @staticmethod
    def _match_function_by_rules(e: EquipmentInput) -> Tuple[Optional[str], float, str]:
        """
        룰 기반 Function Class 매칭.
        return: (function_id, score, explanation)
        """
        text = (e.name + " " + (e.system or "") + " " + (e.description or "")).lower()
        best_id = None
        best_score = 0
        best_rule_name = ""

        for rule in E26_FUNCTION_RULES:
            count = 0
            for kw in rule["keywords"]:
                if kw in text:
                    count += 1
            if count > 0 and count > best_score:
                best_score = count
                best_id = rule["id"]
                best_rule_name = rule["name_en"]

        if best_id is None:
            return None, 0.0, "No rule-based match"
        else:
            return best_id, float(best_score), f"Matched by keywords to {best_rule_name}"

    def _apply_criteria(self, function_id: Optional[str], e: EquipmentInput) -> Tuple[Dict[str, bool], float]:
        """
        Function + 네트워크 정보로 Criteria 매칭 (간단 버전, 일부 추측입니다).
        실제 적용 시 UR E26 + Yard 기준으로 Criteria 정의/가중치 튜닝 필요.
        """
        criteria = {key: False for key in CRITERIA_RULES.keys()}
        score = 0.0

        # Function 기반 Safety/Environment/Operation Critical 추론
        if function_id:
            rule = next((r for r in E26_FUNCTION_RULES if r["id"] == function_id), None)
            if rule:
                crit = rule["default_criticality"]
                if crit.get("safety"):
                    criteria["safety_critical"] = True
                    score += 2.0
                if crit.get("environment"):
                    criteria["environment_critical"] = True
                    score += 1.5
                if crit.get("operation"):
                    criteria["operation_critical"] = True
                    score += 1.0

        # 네트워크 기반 Criteria
        if e.network_zone:
            criteria["networked_with_other_cbs"] = True
            score += 0.5

        if e.connected_to_external_network:
            criteria["connected_to_untrusted"] = True
            score += 1.0

        return criteria, score

    # ----- main classification -----

    def classify(self, e: EquipmentInput) -> E26Decision:
        explanation_lines = []

        # 1) CBS 여부 판단
        is_cbs = self._is_cbs_by_heuristic(e)
        explanation_lines.append(f"[CBS 판단] 휴리스틱 기준으로 is_cbs = {is_cbs}")

        # 2) Function Class (룰 기반)
        rule_func, rule_score, rule_exp = self._match_function_by_rules(e)
        explanation_lines.append(f"[Function-Rule] {rule_exp} (score={rule_score})")

        # 3) Function Class (Naive Bayes 학습 기반)
        text = self._combine_text(e)
        nb_probs = self.nb_model.predict_proba(text)
        if nb_probs:
            nb_best_func = max(nb_probs, key=nb_probs.get)
            nb_best_prob = nb_probs[nb_best_func]
            explanation_lines.append(
                f"[Function-ML] Best={nb_best_func} (p={nb_best_prob:.2f})"
            )
        else:
            nb_best_func, nb_best_prob = None, 0.0
            explanation_lines.append("[Function-ML] No ML knowledge yet")

        # 4) Function 최종 결정 (룰+ML 결합 로직)
        final_func = None
        final_conf = 0.0
        if rule_func and nb_best_func:
            if rule_func == nb_best_func:
                final_func = rule_func
                final_conf = min(1.0, 0.5 + 0.1 * rule_score + 0.4 * nb_best_prob)
                explanation_lines.append(
                    f"[Function-Final] Rule와 ML이 일치 → {final_func}"
                )
            else:
                # 불일치 시 rule vs ML 가중치 비교 (현재 값은 추측입니다)
                if rule_score >= 2 or nb_best_prob < 0.4:
                    final_func = rule_func
                    final_conf = 0.5
                    explanation_lines.append(
                        f"[Function-Final] Rule 우선 적용 (rule={rule_func}, ml={nb_best_func})"
                    )
                else:
                    final_func = nb_best_func
                    final_conf = nb_best_prob
                    explanation_lines.append(
                        f"[Function-Final] ML 우선 적용 (ml={nb_best_func}, rule={rule_func})"
                    )
        elif rule_func:
            final_func = rule_func
            final_conf = min(1.0, 0.3 + 0.1 * rule_score)
            explanation_lines.append(
                f"[Function-Final] Rule만 존재 → {final_func}"
            )
        elif nb_best_func:
            final_func = nb_best_func
            final_conf = nb_best_prob
            explanation_lines.append(
                f"[Function-Final] ML만 존재 → {final_func}"
            )
        else:
            final_func = None
            final_conf = 0.0
            explanation_lines.append(
                "[Function-Final] Function 분류 불가 (정보/학습 부족)"
            )

        # 사용자가 Function 힌트를 줬는데, 내부 판단과 다르면 경고
        warnings = []
        if e.function_hint and final_func:
            hint = e.function_hint.strip().upper()
            if hint != final_func.upper():
                warnings.append(
                    f"입력된 function_hint({e.function_hint})와 내부 판단({final_func})가 다릅니다."
                )

        # 5) Criteria 적용 & E26 대상 여부
                # 5) Criteria 적용 & E26 대상 여부
        criteria_matches, criteria_score = self._apply_criteria(final_func, e)
        base_score = 0.0
        if is_cbs:
            base_score += 1.0
        total_score = base_score + criteria_score

        # 이 임계값은 현재 경험 기반 가정(추측입니다)
        in_scope = is_cbs and total_score >= 2.5
        explanation_lines.append(
            f"[Scope] base_score={base_score:.1f}, criteria_score={criteria_score:.1f}, total={total_score:.1f}, "
            f"in_scope={in_scope}"
        )

        # 먼저 결정 객체를 만들고
        decision = E26Decision(
            is_cbs=is_cbs,
            function_class=final_func,
            in_scope=in_scope,
            criteria_matches=criteria_matches,
            score=total_score,
            confidence=final_conf,
            warnings=warnings,
            explanation="",  # 일단 빈 값
        )

        # 디버깅 로그 대신, 사람이 보기에 쉬운 요약 설명을 넣어준다
        decision.explanation = build_human_explanation(e, decision)

        return decision

    # ----- 사용자 피드백을 통한 학습 -----

    def feedback(
        self,
        e: EquipmentInput,
        correct_function_class: Optional[str] = None,
        correct_is_cbs: Optional[bool] = None,
        correct_in_scope: Optional[bool] = None,
    ):
        """
        사용자가 “이 장비는 실제로는 XXX Function이고, E26 대상/비대상이다” 라고 알려줄 때 호출.
        - NAIVE BAYES는 function_class만 학습 (텍스트 -> Function 매핑 개선)
        - CBS 여부, in_scope 여부는 지금 구조에선 통계만 모으고, 향후 로직 개선에 활용 가능
        """
        text = self._combine_text(e)
        if correct_function_class:
            self.nb_model.update(correct_function_class, text)

        # 필요하다면, CBS 여부 / in_scope 통계도 여기에 기록하는 로직 추가 가능
        # (예: self.cbs_stats[pattern] += 1 같은 형태. 지금은 골격만.)

        # 학습 후 바로 모델 저장
        self.save_model()

def build_human_explanation(
    eq: EquipmentInput,
    decision: E26Decision,
) -> str:
    """
    사람이 보기 쉬운 짧은 요약 설명 생성기.
    - 장비 이름
    - CBS 여부
    - Function 및 신뢰도
    - 어느 Criteria에 걸렸는지
    - 점수와 최종 E26 대상/비대상
    """

    lines = []

    # 0) 장비 이름 요약
    title = eq.name or eq.system or "(이름 없음)"
    lines.append(f"[요약] '{title}' → "
                 f"{'UR E26 대상 장비' if decision.in_scope else 'UR E26 비대상 장비'}")

    # 1) CBS 여부
    lines.append(
        f"- CBS 여부: "
        f"{'CBS (컴퓨터 기반 시스템)' if decision.is_cbs else '비-CBS (단순 장치/패시브 장비 등)'}"
    )

    # 2) Function 분류
    func = decision.function_class or "UNCLASSIFIED"
    lines.append(
        f"- Function: {func} "
        f"(신뢰도 {decision.confidence:.2f})"
    )

    # 3) Criteria 충족 내역
    crit_labels = {
        "safety_critical": "Safety",
        "environment_critical": "Environment",
        "operation_critical": "Operation",
        "networked_with_other_cbs": "Onboard Networked CBS",
        "connected_to_untrusted": "Untrusted/External Network",
    }

    true_crit = [
        label
        for key, label in crit_labels.items()
        if decision.criteria_matches.get(key, False)
    ]

    if true_crit:
        lines.append(
            "- 충족된 E26 Criteria: " + ", ".join(true_crit)
        )
    else:
        lines.append(
            "- 충족된 E26 Criteria: 없음 (핵심 기준에 해당하는 부분이 없음)"
        )

    # 4) 점수와 최종 판정 기준
    lines.append(
        f"- 점수: {decision.score:.1f}점 "
        f"(임계값 2.5 이상 & CBS일 때 E26 대상)"
    )

    # 5) 경고가 있으면 한 줄로 요약
    if decision.warnings:
        # 여러 개여도 한 줄로 이어서
        warn_text = " | ".join(decision.warnings)
        lines.append(f"- 참고 사항: {warn_text}")

    return "\n".join(lines)



# =========================
# 5. 사용 예시
# =========================

if __name__ == "__main__":
    clf = E26Classifier()

    # 예시: 메인 엔진 원격제어 시스템
    eq = EquipmentInput(
        id="EQ-001",
        name="Main Engine Remote Control System",
        vendor="ABC Marine",
        system="Main Engine",
        function_hint="PROPULSION",
        description="ME remote control and monitoring panel, integrated with bridge control system",
        is_computer_based=True,
        controls_physical_process=True,
        network_zone="Propulsion",
        connected_to_external_network=False,
    )

    decision = clf.classify(eq)

    print("=== E26 Classification Result ===")
    print(f"ID: {eq.id}")
    print(f"Name: {eq.name}")
    print(f"- is_cbs: {decision.is_cbs}")
    print(f"- function_class: {decision.function_class}")
    print(f"- in_scope (E26 대상 장비): {decision.in_scope}")
    print(f"- score: {decision.score:.2f}")
    print(f"- confidence: {decision.confidence:.2f}")
    print(f"- criteria:")
    for k, v in decision.criteria_matches.items():
        print(f"    {k}: {v}")
    if decision.warnings:
        print("- warnings:")
        for w in decision.warnings:
            print(f"    {w}")
    print("\n--- explanation ---")
    print(decision.explanation)

    # 예시: 사용자가 실제 Function을 'PROPULSION'이라고 확정해주는 경우
    clf.feedback(
        eq,
        correct_function_class="PROPULSION",
        correct_is_cbs=True,
        correct_in_scope=True
    )
    print("\n[INFO] Feedback applied & model updated.")
