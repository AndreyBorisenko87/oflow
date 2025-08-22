"""
Блок 14: Бэктест
Правила вход/выход/стоп/цель и сводка метрик
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Движок бэктестинга для торговых стратегий"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategy_config = config.get('strategy', {})
        self.risk_config = config.get('risk_management', {})
        self.position_config = config.get('position_sizing', {})
        
        # Настройки стратегии
        self.entry_rules = self.strategy_config.get('entry_rules', {})
        self.exit_rules = self.strategy_config.get('exit_rules', {})
        self.stop_loss = self.risk_config.get('stop_loss_pct', 2.0)
        self.take_profit = self.risk_config.get('take_profit_pct', 4.0)
        self.max_position_size = self.position_config.get('max_size', 1.0)
        
        # Состояние бэктеста
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.trades_history = []
        
        logger.info(f"Бэктест движок инициализирован: SL={self.stop_loss}%, TP={self.take_profit}%")
        
    def run_backtest(self, events_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Запуск полного бэктеста"""
        logger.info("🚀 Запуск бэктеста торговой стратегии...")
        
        if events_df.empty:
            logger.warning("Нет событий для бэктеста")
            return pd.DataFrame()
        
        if quotes_df.empty:
            logger.warning("Нет котировок для бэктеста")
            return pd.DataFrame()
        
        # Сортируем данные по времени
        events_df = events_df.sort_values('ts_ns').reset_index(drop=True)
        quotes_df = quotes_df.sort_values('ts_ns').reset_index(drop=True)
        
        logger.info(f"Обработка {len(events_df)} событий и {len(quotes_df)} котировок")
        
        # Основной цикл бэктеста
        for idx, event in events_df.iterrows():
            if idx % 100 == 0:  # Логируем каждые 100 событий
                logger.info(f"Прогресс бэктеста: {idx+1}/{len(events_df)} событий")
            
            # Получаем текущую цену
            current_price = self._get_current_price(quotes_df, event['ts_ns'])
            if current_price is None:
                continue
            
            # Проверяем правила входа/выхода
            if self.current_position == 0:
                # Нет позиции - проверяем вход
                if self._should_enter(event, current_price):
                    self._enter_position(event, current_price)
            else:
                # Есть позиция - проверяем выход
                if self._should_exit(event, current_price):
                    self._exit_position(event, current_price)
        
        # Закрываем открытую позицию в конце
        if self.current_position != 0:
            self._force_close_position(events_df.iloc[-1], quotes_df.iloc[-1])
        
        # Создаем DataFrame с результатами
        trades_df = pd.DataFrame(self.trades_history)
        
        if not trades_df.empty:
            logger.info(f"✅ Бэктест завершен: {len(trades_df)} сделок")
            logger.info(f"Общий P&L: {trades_df['pnl'].sum():.2f}")
        else:
            logger.warning("⚠️ Бэктест не дал результатов")
        
        return trades_df
    
    def _get_current_price(self, quotes_df: pd.DataFrame, timestamp: int) -> float:
        """Получить текущую цену для заданного времени"""
        # Ищем ближайшую котировку по времени
        time_diff = abs(quotes_df['ts_ns'] - timestamp)
        if time_diff.empty:
            return None
        
        closest_idx = time_diff.idxmin()
        return quotes_df.loc[closest_idx, 'mid']
    
    def _should_enter(self, event: pd.Series, current_price: float) -> bool:
        """Проверка условий для входа в позицию"""
        # Базовые правила входа
        if 'confidence' in event and event['confidence'] < 0.7:
            return False
        
        # Проверка типа паттерна
        pattern_type = event.get('pattern_type', '')
        if pattern_type in ['liquidity_vacuum_break', 'iceberg_fade']:
            return True
        
        # Проверка объема
        if 'volume_ratio' in event and event['volume_ratio'] < 2.0:
            return False
        
        return True
    
    def _should_exit(self, event: pd.Series, current_price: float) -> bool:
        """Проверка условий для выхода из позиции"""
        if self.current_position == 0:
            return False
        
        # Расчет текущего P&L
        if self.current_position > 0:  # Long позиция
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:  # Short позиция
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
        
        # Stop Loss
        if pnl_pct <= -self.stop_loss:
            logger.info(f"🔴 Stop Loss сработал: {pnl_pct:.2f}%")
            return True
        
        # Take Profit
        if pnl_pct >= self.take_profit:
            logger.info(f"🟢 Take Profit сработал: {pnl_pct:.2f}%")
            return True
        
        # Выход по времени (если позиция открыта слишком долго)
        if self.entry_time and 'ts_ns' in event:
            time_in_trade = (event['ts_ns'] - self.entry_time) / 1e9  # в секундах
            if time_in_trade > 300:  # 5 минут
                logger.info(f"⏰ Выход по времени: {time_in_trade:.0f}с")
                return True
        
        return False
    
    def _enter_position(self, event: pd.Series, current_price: float):
        """Вход в позицию"""
        # Определяем направление позиции
        if event.get('pattern_type') in ['liquidity_vacuum_break']:
            position_size = self.max_position_size
        else:
            position_size = self.max_position_size * 0.5
        
        self.current_position = position_size
        self.entry_price = current_price
        self.entry_time = event['ts_ns']
        
        logger.info(f"📈 Вход в позицию: {position_size} @ {current_price:.4f}")
    
    def _exit_position(self, event: pd.Series, current_price: float):
        """Выход из позиции"""
        if self.current_position == 0:
            return
        
        # Расчет P&L
        if self.current_position > 0:  # Long позиция
            pnl = (current_price - self.entry_price) * self.current_position
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
        else:  # Short позиция
            pnl = (self.entry_price - current_price) * abs(self.current_position)
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
        
        # Сохраняем сделку
        trade = {
            'entry_time': pd.to_datetime(self.entry_time, unit='ns'),
            'exit_time': pd.to_datetime(event['ts_ns'], unit='ns'),
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'position_size': self.current_position,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'pattern_type': event.get('pattern_type', ''),
            'confidence': event.get('confidence', 0.0),
            'exchange': event.get('exchange', ''),
            'duration_seconds': (event['ts_ns'] - self.entry_time) / 1e9
        }
        
        self.trades_history.append(trade)
        
        logger.info(f"📉 Выход из позиции: P&L = {pnl:.4f} ({pnl_pct:.2f}%)")
        
        # Сброс состояния
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_time = None
    
    def _force_close_position(self, event: pd.Series, quote: pd.Series):
        """Принудительное закрытие позиции в конце бэктеста"""
        if self.current_position == 0:
            return
        
        current_price = quote.get('mid', self.entry_price)
        self._exit_position(event, current_price)
    
    def calculate_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Расчет комплексных метрик бэктеста"""
        logger.info("📊 Расчет метрик бэктеста...")
        
        if trades_df.empty:
            return {
                "total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "profit_factor": 0.0, "avg_trade": 0.0,
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0
            }
        
        # Базовые метрики
        total_return = trades_df['pnl'].sum()
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # Процент выигрышных сделок
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Средняя прибыль/убыток
        avg_trade = trades_df['pnl'].mean()
        
        # Profit Factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Коэффициент Шарпа (упрощенный)
        returns = trades_df['pnl_pct'] / 100
        sharpe_ratio = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)  # Годовой
        
        # Максимальная просадка
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-8) * 100
        max_drawdown = abs(drawdown.min())
        
        # Дополнительные метрики
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0.0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0.0
        
        # Средняя продолжительность сделки
        avg_duration = trades_df['duration_seconds'].mean() if 'duration_seconds' in trades_df.columns else 0.0
        
        metrics = {
            "total_return": total_return,
            "total_return_pct": trades_df['pnl_pct'].sum(),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "avg_duration": avg_duration
        }
        
        # Логируем основные метрики
        logger.info("=== МЕТРИКИ БЭКТЕСТА ===")
        logger.info(f"💰 Общий доход: {total_return:.4f} ({trades_df['pnl_pct'].sum():.2f}%)")
        logger.info(f"📈 Всего сделок: {total_trades}")
        logger.info(f"✅ Выигрышных: {winning_trades} ({win_rate*100:.1f}%)")
        logger.info(f"❌ Проигрышных: {losing_trades}")
        logger.info(f"📊 Profit Factor: {profit_factor:.2f}")
        logger.info(f"📉 Максимальная просадка: {max_drawdown:.2f}%")
        logger.info(f"🎯 Коэффициент Шарпа: {sharpe_ratio:.2f}")
        
        return metrics

def generate_report(
    trades_df: pd.DataFrame,
    metrics: Dict[str, float],
    output_dir: Path = Path("data/backtest")
) -> None:
    """Генерация комплексного отчета бэктеста"""
    logger.info("📋 Генерация отчета бэктеста...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Сохранение сделок в parquet
    if not trades_df.empty:
        trades_path = output_dir / "backtest_trades.parquet"
        trades_df.to_parquet(trades_path, engine="pyarrow", index=False)
        logger.info(f"✅ Сделки сохранены: {trades_path}")
        
        # 2. Сохранение сделок в CSV для удобства
        csv_path = output_dir / "backtest_trades.csv"
        trades_df.to_csv(csv_path, index=False)
        logger.info(f"✅ CSV отчет создан: {csv_path}")
    
    # 3. Сохранение метрик в JSON
    metrics_path = output_dir / "backtest_metrics.json"
    import json
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"✅ Метрики сохранены: {metrics_path}")
    
    # 4. Создание сводного отчета
    summary_path = output_dir / "backtest_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== ОТЧЕТ БЭКТЕСТА ===\n\n")
        f.write(f"Дата: {pd.Timestamp.now()}\n")
        f.write(f"Всего сделок: {metrics.get('total_trades', 0)}\n")
        f.write(f"Выигрышных: {metrics.get('winning_trades', 0)}\n")
        f.write(f"Проигрышных: {metrics.get('losing_trades', 0)}\n")
        f.write(f"Процент выигрышных: {metrics.get('win_rate', 0)*100:.1f}%\n\n")
        
        f.write("=== МЕТРИКИ ===\n")
        f.write(f"Общий доход: {metrics.get('total_return', 0):.4f}\n")
        f.write(f"Общий доход (%): {metrics.get('total_return_pct', 0):.2f}%\n")
        f.write(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n")
        f.write(f"Коэффициент Шарпа: {metrics.get('sharpe_ratio', 0):.2f}\n")
        f.write(f"Максимальная просадка: {metrics.get('max_drawdown', 0):.2f}%\n")
        f.write(f"Средняя сделка: {metrics.get('avg_trade', 0):.4f}\n")
        f.write(f"Средняя выигрышная: {metrics.get('avg_win', 0):.4f}\n")
        f.write(f"Средняя проигрышная: {metrics.get('avg_loss', 0):.4f}\n")
        f.write(f"Средняя продолжительность: {metrics.get('avg_duration', 0):.1f}с\n")
    
    logger.info(f"✅ Сводный отчет создан: {summary_path}")
    logger.info(f"📁 Все файлы сохранены в: {output_dir}")

def run_backtest(
    events_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    config: Dict,
    output_dir: Path = Path("data/backtest")
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Основная функция бэктеста торговой стратегии"""
    logger.info("🚀 Запуск бэктеста торговой стратегии...")
    
    try:
        # 1. Создать движок бэктеста
        logger.info("Создание движка бэктеста...")
        engine = BacktestEngine(config)
        
        # 2. Запустить основной бэктест
        logger.info("Запуск основного бэктеста...")
        trades_df = engine.run_backtest(events_df, quotes_df)
        
        if trades_df.empty:
            logger.warning("⚠️ Бэктест не дал результатов")
            # Создаем пустые метрики
            metrics = engine.calculate_metrics(trades_df)
        else:
            # 3. Рассчитать комплексные метрики
            logger.info("Расчет метрик бэктеста...")
            metrics = engine.calculate_metrics(trades_df)
        
        # 4. Сгенерировать отчеты
        logger.info("Генерация отчетов...")
        generate_report(trades_df, metrics, output_dir)
        
        # 5. Итоговая статистика
        logger.info("=== ИТОГИ БЭКТЕСТА ===")
        if not trades_df.empty:
            logger.info(f"📊 Всего сделок: {len(trades_df)}")
            logger.info(f"💰 Общий доход: {metrics.get('total_return', 0):.4f}")
            logger.info(f"📈 Процент выигрышных: {metrics.get('win_rate', 0)*100:.1f}%")
            logger.info(f"🎯 Коэффициент Шарпа: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"📉 Максимальная просадка: {metrics.get('max_drawdown', 0):.2f}%")
        else:
            logger.info("📊 Результатов нет")
        
        logger.info("✅ Бэктест успешно завершен!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при выполнении бэктеста: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return trades_df, metrics
