# 🚀 **БЫСТРЫЙ СТАРТ - SCALPING STRATEGY**

## 🌿 **СТРУКТУРА ВЕТОК**

```
main          ← Продакшн код (стабильные релизы)
develop       ← Разработка (интеграция)
feature/*     ← Новые функции
hotfix/*      ← Срочные исправления
```

## ⚡ **БЫСТРЫЕ КОМАНДЫ**

### **1. Создание feature ветки**
```bash
# Переключиться на develop
git checkout develop

# Создать feature ветку
git checkout -b feature/название-функции

# Или одной командой
git flow feature start название-функции
```

### **2. Работа в feature ветке**
```bash
# Добавить изменения
git add .

# Закоммитить
git commit -m "Описание изменений"

# Синхронизироваться с develop
git checkout develop
git pull origin develop
git checkout feature/название-функции
git rebase develop
```

### **3. Завершение feature**
```bash
# Переключиться на develop
git checkout develop

# Мерж feature ветки
git merge feature/название-функции

# Удалить feature ветку
git branch -d feature/название-функции

# Или Git Flow командой
git flow feature finish название-функции
```

### **4. Создание hotfix**
```bash
# Переключиться на main
git checkout main

# Создать hotfix ветку
git checkout -b hotfix/описание-проблемы

# Или Git Flow командой
git flow hotfix start описание-проблемы
```

### **5. Завершение hotfix**
```bash
# Мерж в main
git checkout main
git merge hotfix/описание-проблемы

# Мерж в develop
git checkout develop
git merge hotfix/описание-проблемы

# Удалить hotfix ветку
git branch -d hotfix/описание-проблемы

# Или Git Flow командой
git flow hotfix finish описание-проблемы
```

## 📝 **ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ**

### **Пример 1: Новая функция**
```bash
# Создать ветку для нового блока
git flow feature start block-16-ml-predictions

# Работать над функцией...
git add .
git commit -m "Добавлен ML детектор для предсказания паттернов"

# Завершить feature
git flow feature finish block-16-ml-predictions
```

### **Пример 2: Исправление бага**
```bash
# Создать hotfix для критического бага
git flow hotfix start critical-memory-leak

# Исправить баг...
git add .
git commit -m "Исправлена утечка памяти в детекторе"

# Завершить hotfix
git flow hotfix finish critical-memory-leak
```

## 🔧 **ПОЛЕЗНЫЕ АЛИАСЫ**

### **Настройка алиасов**
```bash
# Создание feature ветки
git config --global alias.feature '!git checkout -b feature/$1 develop'

# Создание hotfix ветки
git config --global alias.hotfix '!git checkout -b hotfix/$1 main'

# Синхронизация с develop
git config --global alias.sync '!git fetch origin && git rebase origin/develop'
```

### **Использование алиасов**
```bash
# Создать feature ветку
git feature новая-функция

# Создать hotfix ветку
git hotfix исправление-бага

# Синхронизироваться
git sync
```

## 📋 **ПРАВИЛА ИМЕНОВАНИЯ**

### **Feature ветки**
```
feature/block-16-ml-predictions      # Новый блок
feature/enhancement-detectors        # Улучшение детекторов
feature/config-optimization          # Оптимизация конфигурации
feature/docs-api-reference           # Документация API
feature/bugfix-memory-leak          # Исправление бага
feature/performance-improvement      # Улучшение производительности
```

### **Hotfix ветки**
```
hotfix/critical-memory-leak         # Критическая утечка памяти
hotfix/security-vulnerability       # Уязвимость безопасности
hotfix/data-corruption              # Повреждение данных
hotfix/api-timeout                  # Таймаут API
```

## 🚨 **ЧТО ДЕЛАТЬ ЕСЛИ...**

### **Конфликт при мерже**
```bash
# Разрешить конфликты в файлах
# Затем добавить и закоммитить
git add .
git commit -m "Разрешен конфликт мержа"
```

### **Нужно отменить последний коммит**
```bash
# Отменить последний коммит (сохранить изменения)
git reset --soft HEAD~1

# Отменить последний коммит (удалить изменения)
git reset --hard HEAD~1
```

### **Нужно переименовать ветку**
```bash
# Переименовать текущую ветку
git branch -m новое-название

# Переименовать другую ветку
git branch -m старое-название новое-название
```

## 📊 **ПРОВЕРКА СТАТУСА**

### **Основные команды**
```bash
# Статус рабочей директории
git status

# Список веток
git branch

# Список всех веток (включая удаленные)
git branch -a

# История коммитов
git log --oneline -10
```

### **Полезные флаги**
```bash
# Показать ветки с последними коммитами
git branch -v

# Показать мерженные ветки
git branch --merged

# Показать немерженные ветки
git branch --no-merged
```

## 🎯 **ЧЕКЛИСТ ДЛЯ РЕЛИЗА**

### **Перед мержем в main**
- [ ] Все тесты проходят
- [ ] Код прошел review
- [ ] Документация обновлена
- [ ] Версия обновлена
- [ ] Changelog создан

### **После мержа в main**
- [ ] Создан тег версии
- [ ] Деплой в продакшн
- [ ] Уведомления отправлены
- [ ] Мониторинг запущен

---

**Версия**: 1.0.0  
**Последнее обновление**: Август 2025  
**Статус**: ✅ Активна
