// DeformableTarget.cs の修正版

using UnityEngine;
using System.Collections.Generic;

public class DeformableTarget : MonoBehaviour
{
    [Header("変形設定")]
    [Range(0f, 1f)]
    public float softness = 0.5f;
    public float maxDeformation = 0.3f;
    public float deformationSpeed = 2f;
    public bool enableVisualDeformation = true;
    
    [Header("物理特性")]
    public float breakingForce = 50f;
    public float compressionResistance = 15f;
    
    [Header("視覚設定")]
    public Color originalColor = Color.white;
    
    [Header("デバッグ")]
    public bool enableDebugLogs = false;
    
    // 内部状態
    private Vector3 originalScale;
    private float currentDeformation = 0f;
    private bool isBroken = false;
    private Renderer objectRenderer;
    private bool isBeingGrasped = false;
    private float appliedForce = 0f;
    
    // 新しい変形システム用
    private Vector3 lastContactNormal = Vector3.zero;
    private List<ContactInfo> activeContacts = new List<ContactInfo>();
    
    [System.Serializable]
    public struct ContactInfo
    {
        public Vector3 position;
        public Vector3 normal;
        public float force;
        public float timestamp;
    }
    
    void Start()
    {
        originalScale = transform.localScale;
        objectRenderer = GetComponent<Renderer>();
        if (objectRenderer != null)
            originalColor = objectRenderer.material.color;
        
        Debug.Log("DeformableTarget initialized");
    }
    
    void Update()
    {
        CleanupOldContacts();
        UpdateAdaptiveDeformation();
        UpdateVisualFeedback();
    }
    
    public void ApplyGripperForce(float force, Vector3 contactPosition)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        
        if (force > breakingForce)
        {
            isBroken = true;
            Debug.Log($"Object broken! Force: {force:F2}N");
        }
        
        CalculateDeformation(force);
        
        if (enableDebugLogs)
            Debug.Log($"Force applied: {force:F2}N, Deformation: {currentDeformation:F3}");
    }
            
    // DeformableTarget.cs も修正（力伝達時の最初の1回のみログ）
    public void ApplyGripperForceWithDirection(float force, Vector3 contactPosition, Vector3 contactNormal)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        
        if (force > breakingForce)
        {
            isBroken = true;
            Debug.Log($"Object broken! Force: {force:F2}N");
        }
        
        // 接触方向の成分を分析（ログは削除）
        float xComponent = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.right));
        float yComponent = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.up));
        float zComponent = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.forward));
        
        // 力伝達検出時に一度だけログ出力
        static bool hasLoggedDirection = false;
        if (!hasLoggedDirection)
        {
            Debug.Log($"=== DEFORMATION DIRECTION ANALYSIS ===");
            Debug.Log($"Force: {force:F2}N, Normal: {contactNormal}");
            Debug.Log($"Direction components - X: {xComponent:F3}, Y: {yComponent:F3}, Z: {zComponent:F3}");
            hasLoggedDirection = true;
        }
        
        // X方向（左右）からの力の場合
        if (xComponent > 0.2f)
        {
            if (!hasLoggedDirection)
            {
                Debug.Log("LEFT-RIGHT COMPRESSION DETECTED!");
            }
            
            // X軸を圧縮、Y・Z軸を拡張する変形を直接適用
            float deformAmount = Mathf.Clamp01(force / compressionResistance) * softness * 0.5f;
            Vector3 newScale = new Vector3(
                originalScale.x * (1f - deformAmount), // X軸圧縮
                originalScale.y * (1f + deformAmount * 0.3f), // Y軸拡張
                originalScale.z * (1f + deformAmount * 0.3f)  // Z軸拡張
            );
            transform.localScale = newScale;
            
            if (!hasLoggedDirection)
            {
                Debug.Log($"Applied LEFT-RIGHT compression scale: {newScale}");
            }
        }
        else
        {
            // 従来の変形
            CalculateDirectionalDeformation(force, contactNormal);
            
            if (!hasLoggedDirection)
            {
                Debug.Log("Using traditional Y-axis compression");
            }
        }
    }
    
    private void RecordContact(Vector3 position, Vector3 normal, float force)
    {
        // 新しい接触情報を追加
        ContactInfo contact = new ContactInfo
        {
            position = position,
            normal = normal,
            force = force,
            timestamp = Time.time
        };
        
        activeContacts.Add(contact);
        lastContactNormal = normal;
        
        // 古い接触情報を制限（最新10個まで）
        if (activeContacts.Count > 10)
        {
            activeContacts.RemoveAt(0);
        }
    }
    
    private void CleanupOldContacts()
    {
        // 1秒以上古い接触情報を削除
        activeContacts.RemoveAll(contact => Time.time - contact.timestamp > 1f);
    }
    
    private void CalculateDeformation(float force)
    {
        float targetDeformation = Mathf.Clamp01(force / compressionResistance) * softness;
        targetDeformation = Mathf.Clamp(targetDeformation, 0f, maxDeformation);
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, deformationSpeed * Time.deltaTime);
    }
    
    private void CalculateDirectionalDeformation(float force, Vector3 contactNormal)
    {
        float baseDeformation = Mathf.Clamp01(force / compressionResistance) * softness;
        
        // 接触方向に基づく変形係数の計算
        float directionFactor = CalculateDirectionFactor(contactNormal);
        
        float targetDeformation = baseDeformation * directionFactor;
        targetDeformation = Mathf.Clamp(targetDeformation, 0f, maxDeformation);
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, deformationSpeed * Time.deltaTime);
    }
    
    private float CalculateDirectionFactor(Vector3 contactNormal)
    {
        // Y軸（上下）方向の場合は通常の変形
        float yComponent = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.up));
        
        // X軸またはZ軸（左右・前後）方向の場合は側面変形
        float xComponent = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.right));
        float zComponent = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.forward));
        float sideComponent = Mathf.Max(xComponent, zComponent);
        
        // 側面からの力により強く反応するように調整
        float directionFactor = Mathf.Lerp(1f, 0.7f, yComponent) + (sideComponent * 0.5f);
        
        return Mathf.Clamp(directionFactor, 0.3f, 1.5f);
    }
    
    private void UpdateAdaptiveDeformation()
    {
        if (!enableVisualDeformation || isBroken) return;
        
        if (!isBeingGrasped)
        {
            // 把持されていない場合は元の形に戻る
            currentDeformation = Mathf.Lerp(currentDeformation, 0f, deformationSpeed * Time.deltaTime);
            transform.localScale = Vector3.Lerp(transform.localScale, originalScale, Time.deltaTime * deformationSpeed);
            return;
        }
        
        // 接触方向に基づいた適応的変形
        Vector3 deformedScale = CalculateAdaptiveScale();
        transform.localScale = Vector3.Lerp(transform.localScale, deformedScale, Time.deltaTime * deformationSpeed);
    }
       

    private Vector3 CalculateAdaptiveScale()
    {
        if (activeContacts.Count == 0)
        {
            return CalculateDefaultDeformation();
        }
        
        // 複数の接触点から主要な変形方向を決定
        Vector3 avgContactNormal = Vector3.zero;
        float totalForce = 0f;
        
        foreach (var contact in activeContacts)
        {
            avgContactNormal += contact.normal * contact.force;
            totalForce += contact.force;
        }
        
        if (totalForce > 0)
        {
            avgContactNormal = (avgContactNormal / totalForce).normalized;
        }
        
        // 接触方向の成分を分析
        float xComponent = Mathf.Abs(Vector3.Dot(avgContactNormal, Vector3.right));
        float yComponent = Mathf.Abs(Vector3.Dot(avgContactNormal, Vector3.up));
        float zComponent = Mathf.Abs(Vector3.Dot(avgContactNormal, Vector3.forward));
        
        Vector3 deformedScale = originalScale;
        float deformationAmount = currentDeformation;
        
        // デバッグ情報
        if (enableDebugLogs)
        {
            Debug.Log($"Contact Analysis - Normal: {avgContactNormal}, " +
                    $"Components: X={xComponent:F2}, Y={yComponent:F2}, Z={zComponent:F2}, " +
                    $"Contacts: {activeContacts.Count}");
        }
        
        // 左右からの把持（X軸方向の圧力）
        if (xComponent > 0.5f)
        {
            deformedScale = new Vector3(
                originalScale.x * (1f - deformationAmount * 0.6f), // X軸を圧縮
                originalScale.y * (1f + deformationAmount * 0.2f), // Y軸を拡張
                originalScale.z * (1f + deformationAmount * 0.3f)  // Z軸を拡張
            );
            
            if (enableDebugLogs)
                Debug.Log("Left-Right compression applied");
        }
        // 前後からの把持（Z軸方向の圧力）
        else if (zComponent > 0.5f)
        {
            deformedScale = new Vector3(
                originalScale.x * (1f + deformationAmount * 0.3f), // X軸を拡張
                originalScale.y * (1f + deformationAmount * 0.2f), // Y軸を拡張
                originalScale.z * (1f - deformationAmount * 0.6f)  // Z軸を圧縮
            );
            
            if (enableDebugLogs)
                Debug.Log("Front-Back compression applied");
        }
        // 上下からの把持（Y軸方向の圧力）
        else if (yComponent > 0.5f)
        {
            deformedScale = CalculateDefaultDeformation();
            
            if (enableDebugLogs)
                Debug.Log("Top-Bottom compression applied");
        }
        // 複合方向または不明確な場合
        else
        {
            // より強い成分の方向を優先
            if (xComponent >= yComponent && xComponent >= zComponent)
            {
                // X方向優先
                deformedScale = new Vector3(
                    originalScale.x * (1f - deformationAmount * 0.4f),
                    originalScale.y * (1f + deformationAmount * 0.15f),
                    originalScale.z * (1f + deformationAmount * 0.2f)
                );
            }
            else if (zComponent >= yComponent)
            {
                // Z方向優先
                deformedScale = new Vector3(
                    originalScale.x * (1f + deformationAmount * 0.2f),
                    originalScale.y * (1f + deformationAmount * 0.15f),
                    originalScale.z * (1f - deformationAmount * 0.4f)
                );
            }
            else
            {
                // Y方向優先（従来の動作）
                deformedScale = CalculateDefaultDeformation();
            }
            
            if (enableDebugLogs)
                Debug.Log("Mixed direction compression applied");
        }
        
        return deformedScale;
    }
    
    private Vector3 CalculateDefaultDeformation()
    {
        // 従来の上下圧縮変形
        float compressionFactor = 1f - currentDeformation;
        return new Vector3(
            originalScale.x * (1f + currentDeformation * 0.2f),
            originalScale.y * compressionFactor,
            originalScale.z * (1f + currentDeformation * 0.2f)
        );
    }
    
    private void UpdateVisualFeedback()
    {
        if (objectRenderer == null) return;
        
        Color stressColor = Color.Lerp(originalColor, Color.red, currentDeformation);
        objectRenderer.material.color = stressColor;
        
        if (isBroken)
        {
            objectRenderer.material.color = Color.gray;
        }
    }
    

        public ObjectState GetCurrentState()
        {
            return new ObjectState
            {
                deformation = currentDeformation,
                appliedForce = appliedForce,
                isBroken = isBroken,
                isBeingGrasped = isBeingGrasped,
                materialType = 1, // Medium - BasicTypes.cs の int materialType に合わせる
                softness = softness
            };
        }
        
        public void ResetObject()
        {
            isBroken = false;
            currentDeformation = 0f;
            appliedForce = 0f;
            isBeingGrasped = false;
            activeContacts.Clear();
            lastContactNormal = Vector3.zero;
            transform.localScale = originalScale;
            
            if (objectRenderer != null)
                objectRenderer.material.color = originalColor;
            
            Debug.Log("Object reset");
        }
        
    }